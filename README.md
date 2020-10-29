# C-COMA
동적 환경에서의 지속적인 다중 에이전트 강화 학습

# 스타크래프트2 멀티 에이전트 챌린지
본 자료는 Windows 운영체제에서 편리하게 실행하기 위해 편집한 가이드.

가장 먼저 스타크래프트2 게임을 설치 하여야 합니다. 체험판도 상관 없습니다. 아래 링크에서 다운 받으세요

https://starcraft2.com/ko-kr/

설치 후 아래 링크에서 미니게임에 필요한 맵을 다운로드 받아야 합니다.

https://github.com/oxwhirl/smac/tree/master/smac/env/starcraft2/maps/SMAC_Maps

다운 받은 파일을 아래 경로에 모두 옮겨 주시면 됩니다.

C:\Program Files (x86)\StarCraft II\Maps\SMAC_Maps

본 논문에서 사용한 동적 환경은 DynamicMaps 내부에 있습니다. 이 파일들도 상위 경로에 넣어 주시면 됩니다.
Training Env : Dynamic_env_Training.SC2Map
Testing Env : Dynamic_env_Test1.SC2Map


# 이제부터는 환경 설정입니다.
우선적으로 필요한 패키지 설치를 위해 아래와 같이 명령을 넣어 주세요
 ```shell
pip install -r requirements.txt
```
안타깝게도 아래 두가지는 직접 설치 해야 합니다.(어렵지 않습니다.)

cloudpickle을 설치 해야 합니다.
 ```shell
pip install cloudpickle
```
pytorch도 설치해야 합니다.
 ```shell
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
```
마지막으로 main.py를 실행 하시면 됩니다.

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
