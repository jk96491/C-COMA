# C-COMA
<b>논문번호 : KIPS_C2020B0262</b>

동적 환경에서의 지속적인 다중 에이전트 강화 학습(Continual Multi-agent Reinforcement Learning in Dynamic Environments)

<b> <font size="200">Architecture</font></b>

<img src="https://user-images.githubusercontent.com/17878413/97534231-f9e63a00-19fc-11eb-8f9e-397e3061c777.png" width="90%"></img>

<b>Dynamic environment Image</b> 

<img src="https://user-images.githubusercontent.com/17878413/97533656-f56d5180-19fb-11eb-8f8c-3d4d468fa1f9.png" width="50%"></img>

<img src="https://user-images.githubusercontent.com/17878413/97533679-fa320580-19fb-11eb-9db1-10e7f6169bfb.png" width="50%"></img>


# StarCraft II Multi Agent Challenge
This repository is a guide edited for convenient execution in the Windows OS.

First you need to install the StarCraft 2 game. Trial version does not matter. Download it from the link below

```shell
https://starcraft2.com/ko-kr/
```

After installation, you should download the map required for the minigame from the link below.

```shell
https://github.com/oxwhirl/smac/tree/master/smac/env/starcraft2/maps/SMAC_Maps
```

You can move all downloaded files to the path below.

```shell
"C:\Program Files (x86)\StarCraft II\Maps\SMAC_Maps"
```

The dynamic environment used in this paper is inside DynamicMaps. You can put these files in the upper path.

```shell
Training Env : Dynamic_env_Training.SC2Map

Testing Env : Dynamic_env_Test1.SC2Map
```

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.

## Documentation/Support

Documentation is a little sparse at the moment (but will improve!). Please raise an issue in this repo, or email [Tabish](mailto:tabish.rashid@cs.ox.ac.uk)

## Citing PyMARL 

If you use PyMARL in your research, please cite the [SMAC paper](https://arxiv.org/abs/1902.04043).

*M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019.*

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

## License

Code licensed under the Apache License v2.0
