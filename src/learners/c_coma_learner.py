import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.c_coma import C_COMACritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop


class C_COMALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = C_COMACritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.agent_params_test = list(mac.test_parameters())
        self.critic_train_params = list(self.critic.get_traing_prams())
        self.critic_test_params = list(self.critic.get_test_prams())
        self.params = self.agent_params + self.critic_train_params + self.critic_test_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.agent_optimiser_test = RMSprop(params=self.agent_params_test, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_train_optimiser = RMSprop(params=self.critic_train_params, lr=args.critic_lr,
                                              alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_test_optimiser = RMSprop(params=self.critic_train_params, lr=args.critic_lr,
                                              alpha=args.optim_alpha, eps=args.optim_eps)

        self.use_observation_critic = False
        self.execution_training_mode = False
        self.train_critic = False

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        q_vals, critic_train_stats = self._train_critic(batch, rewards, terminated, actions, avail_actions,
                                                        critic_mask, bs, max_t)

        actions = actions[:,:-1]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        # Calculated baseline
        q_vals = q_vals.reshape(-1, self.n_actions)
        pi = mac_out.view(-1, self.n_actions)
        baseline = (pi * q_vals).sum(-1).detach()

        # Calculate policy grad with mask
        q_taken = th.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        advantages = (q_taken - baseline).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        coma_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()

        # Optimise agents
        if self.args.Is_continuous_learning:
            optimizer = self.agent_optimiser_test
        else:
            optimizer = self.agent_optimiser

        optimizer.zero_grad()
        coma_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        optimizer.step()

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["td_error_abs", "critic_loss", "critic_grad_norm",  "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):
        # Optimise critic
        target_q_vals_train = self.target_critic(batch)
        target_q_vals_train = target_q_vals_train[:, :]

        targets_taken_train = th.gather(target_q_vals_train, dim=3, index=actions).squeeze(3)

        # Calculate td-lambda targets
        targets_train = build_td_lambda_targets(rewards, terminated, mask, targets_taken_train, self.n_agents, self.args.gamma, self.args.td_lambda)

        q_vals_train = th.zeros_like(target_q_vals_train)[:, :-1]

        running_log = {
            "td_error_abs": [],
            "critic_loss": [],
            "critic_grad_norm": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        for t in reversed(range(rewards.size(1))):
            mask_t = mask[:, t].expand(-1, self.n_agents)
            if mask_t.sum() == 0:
                continue

            q_t_train = self.critic(batch, t)

            # train 학습
            q_vals_train[:, t] = q_t_train.view(bs, self.n_agents, self.n_actions)
            q_taken_train = th.gather(q_t_train, dim=3, index=actions[:, t:t+1]).squeeze(3).squeeze(1)
            targets_t_train = targets_train[:, t]

            td_error_train = (q_taken_train - targets_t_train.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error_train * mask_t

            # Normal L2 loss, take mean over actual data
            train_loss = (masked_td_error ** 2).sum() / mask_t.sum()

            if self.args.Is_continuous_learning:
                optimizer = self.critic_test_optimiser
            else:
                optimizer = self.critic_train_optimiser

            optimizer.zero_grad()
            train_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_train_params, self.args.grad_norm_clip)
            optimizer.step()

            self.critic_training_steps += 1

            running_log["critic_loss"].append(train_loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["q_taken_mean"].append((q_taken_train * mask_t).sum().item() / mask_elems)
            running_log["target_mean"].append((targets_t_train * mask_t).sum().item() / mask_elems)

        return q_vals_train, running_log

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_train_optimiser.state_dict(), "{}/critic_opt_train.th".format(path))
        th.save(self.critic_test_optimiser.state_dict(), "{}/critic_opt_test.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_train_optimiser.load_state_dict(th.load("{}/critic_opt_train.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_test_optimiser.load_state_dict(
            th.load("{}/critic_opt_test.th".format(path), map_location=lambda storage, loc: storage))