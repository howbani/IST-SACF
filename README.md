# IST_SACF

'IST_SACF' is a CARLA-based visual reinforcement learning project for safe car-following. The project focuses on single-lane longitudinal car-following. It uses front-view camera images as the main observation input and trains an agent through frame stacking, image augmentation, experience replay, and a contrastive learning auxiliary objective, enabling the agent to learn a stable time-headway control policy while maintaining safety.

The default training agent in the current repository is cl_cpc, namely a SAC agent equipped with contrastive learning and a predictive CPC branch. The repository also keeps baseline or degraded configurations such as td3, ddpg, and pixel_sac, which are useful for ablation studies and comparative experiments.

## Project Features

- Task type: pure longitudinal car-following. The ego vehicle is controlled by reinforcement learning for longitudinal actions, while lateral control is handled by the environment.
- Training objective: keep THW within the range [1.5, 2.5] under safe TTC conditions.
- Observation input: front-view RGB images, with 3 stacked frames by default.
- Default algorithm: cl_cpc, which includes SAC, a CURL-style contrastive objective, and a predictive CPC branch.
- Optional algorithms: td3, ddpg, and pixel_sac, where contrastive learning is disabled.
- Data augmentation: currently only supports color_jiggle, gaussian_noise, and color_jiggle_and_gaussian_noise.
- Automated execution: the training script automatically starts the CARLA server, so there is no need to launch the simulator manually in advance.
- Evaluation outputs: supports TensorBoard logs, step-by-step CSV statistics, and evaluation video export.

## Repository Structure

```text
IST_SACF/
|-- README.md
|-- environment.yml
|-- train.py
|-- eval.py
|-- cl_cpc.py
|-- carla_env.py
|-- carla_handler.py
|-- carla_augmentations.py
|-- encoder.py
|-- logger.py
|-- settings.py
|-- utils.py
```

The roles of each file are as follows:

- 'train.py': training entry point, responsible for parameter parsing, environment creation, training loop, periodic evaluation, and model saving.
- 'eval.py': offline evaluation entry point, reads in 'args.json' and model weights from the experiment directory, and outputs evaluation metrics.
- 'cl_cpc.py': default agent implementation, including CurlSacAgent, TD3Agent, and DDPGAgent.
- 'carla_env.py': CARLA following environment, defines rewards, termination conditions, observations, and safety statistics.
- 'carla_handler.py': responsible for automatic CARLA server startup, connection, and shutdown. 
- 'carla_augmentations.py': implements augmentation functions for images, and provides a factory for creating augmentors. 
- 'settings.py': global environment configuration for the project, including map, weather, action range, and debug switches.
- 'utils.py': experience replay, frame stacking, random seed, and various training auxiliary functions. 

## Environment Setup

### 1. Basic Requirements
- Operating System: Windows or Linux
- GPU: Recommended to use NVIDIA GPU
- Python: 'python=3.7'
- CARLA: The current configuration uses Town04 as the default scenario, and the spawn parameters in settings.py were mainly tested with CARLA 0.9.14

### 2. Create Conda Environment

The project dependencies are already specified in environment.yml. It is recommended to create an isolated environment directly:

```bash
conda env create -f environment.yml
conda activate IST_SACF
```

The core dependencies included in the environment file are:

- `pytorch==1.13.1`
- `torchvision==0.14.1`
- `torchaudio==0.13.1`
- `pytorch-cuda=11.7`
- `gymnasium`
- `opencv-python`
- `tensorboard`
- `kornia`
- `psutil`

### 3. Install the CARLA Python API

First install and extract CARLA locally, then register the corresponding Python API into the current Conda environment. A common approach is as follows:

```bash
conda activate IST_SACF
conda install -y conda-build
conda develop path/to/CARLA/PythonAPI/carla/dist/carla-<your_version>.egg
```

### 4. Setup 'CARLA_ROOT'

The training script automatically starts the CARLA server through carla_handler.py, so the CARLA root directory environment variable CARLA_ROOT must be set in advance.

Example for Windows PowerShell:

```powershell
$env:CARLA_ROOT="D:\CARLA_0.9.14"
```


## Task Description

This project is not a general-purpose autonomous driving training framework for large-scale scenarios. Instead, it is specifically designed for a single car-following task.

- Map: Town04 by default
- Task: the ego vehicle performs safe car-following when a leading vehicle is present
- Control: reinforcement learning is responsible for longitudinal actions; lateral control is not the main learning target
- Warm-up: before each episode, a fixed warm-up action is applied for 2.5s
- Key metrics: THW, TTC, velocity difference, smoothness, and FiT

The definition of FiT in carla_env.py is as follows:

- 0：running or no special termination
- 1: collision occurred
- 2: time limit reached
- 3: vehicle remained stationary for too long


## Rewards and Safety Objectives

The current reward design focuses on safe car-following. The goal is not simply to maximize speed, but to teach the agent to maintain a reasonable THW under safe TTC conditions.

The main factors considered include:

- Whether THW falls within the target range [1.5, 2.5]
- Whether TTC enters a dangerous range
- Whether the velocity difference is within a safe reasonable range
- Whether the agent has encountered a close encounter with a leading leader
- Whether the agent's control trend is in line with safe scenarios

During training and evaluation, a large number of safety-related statistics are recorded, such as:

- `THW_mean`
- `THW_p50`
- `THW_p95`
- `p_ttc_safe`
- `p_ttc_danger`
- `MAE_dv`
- `FiT`

## Training

### 1. Default Training

Run the following command in the project root directory:

```bash
conda activate IST_SACF
python train.py
```

The default configuration includes:

- agent：`cl_cpc`
- augmentation：`color_jiggle_and_gaussian_noise`
- frame stack：`3`
- batch size：`256`
- train steps：`300000`
- eval frequency：`25000`

### 2. Common Training Commands

Use the default cl_cpc agent:

```bash
python train.py --agent cl_cpc
```

Disable predictive CPC:

```bash
python train.py --agent cl_cpc --no_predictive_cpc
```

Use TD3：

```bash
python train.py --agent td3
```

Use DDPG：

```bash
python train.py --agent ddpg
```

Use Pixel SAC mode:

```bash
python train.py --pixel_sac
```

Change image augmentation:

```bash
python train.py --augmentation color_jiggle
python train.py --augmentation gaussian_noise
python train.py --augmentation color_jiggle_and_gaussian_noise
```

Adjust training steps and evaluation frequency:

```bash
python train.py --num_train_steps 300000 --eval_freq 10000
```

### 3. Common Arguments

The training script currently supports many arguments. The most important ones include:

- Environment:`--carla_town`,`--max_npc_vehicles`,`--seconds_per_episode`,`--fps`
- Camera:`--camera_image_height`,`--camera_image_width`,`--fov`,`--cam_x/y/z`
- Reward weights:`--lambda_r1` to `--lambda_r5`
- Safety thresholds:`--thw_good_lo`,`--thw_good_hi`,`--thw_danger`
- Training hyperparameters:`--batch_size`,`--hidden_dim`,`--discount`
- Encoder parameters:`--encoder_feature_dim`,`--encoder_lr`,`--num_layers`,`--num_filters`
- Runtime controls:`--save_model`,`--save_video`,`--save_buffer`,`--log_interval`
- Debug items:`--cuda_blocking_debug`,`--skip_update_on_error`


## Training Outputs

Each training run creates a new experiment folder under the directory specified by --work_dir_name. The default root directory is experiments.

Output structure is as follows:

```text
experiments/
|-- <experiment_name>/
    |-- args.json
    |-- model/
    |-- buffer/
```

其中：

- 'args.json': saves the complete arguments of the current experiment 
- 'model/':stores the actor, critic, CURL encoder, and predictive CPC-related weights
- 'buffer/':optionally stores experience replay buffer shards

## Evaluation

### 1. Basic Evaluation Command

```bash
python eval.py --experiment_dir_path path/to/your/experiment --model_step 500000
```

Notes:

- `--experiment_dir_path`:the training output experiment directory, which must contain args.json and model/
- `--model_step`:the model checkpoint step to load

The evaluation script automatically:

reads the saved args.json from training
reconstructs the environment and agent using the same observation shape and augmentation configuration
loads the model weights corresponding to the specified step from model/
runs 50 episodes by default for statistics

### 2. Evaluation Weather Settings
`eval.py` uses ClearNoon, ClearSunset, CloudyNoon, CloudySunset, MidRainSunset, WetNoon and WetSunset.

### 3. Evaluation Weather Settings

The evaluation script generates the following contents under the experiment directory:

- `eval_tb_logs/model_<step>/`: evaluation logs
- `eval_detailed_logs/model_<step>/velocity.csv`
- `eval_detailed_logs/model_<step>/ttc.csv`
- `eval_detailed_logs/model_<step>/thw.csv`
- `eval_detailed_logs/model_<step>/smoothness.csv`

During evaluation, the following metrics are computed and printed:

- episode reward
- episode length
- `FiT`
- `pTHW[1.5,2.5]`
- `pTHW[1.0,4.0]`
- `pTTC<2.51`
- `MeanTHW`
- `THW_RMSE`

## Currently Supported Augmentation Methods

The current implementation of `carla_augmentations.py` only supports the following three augmentation methods:

- `color_jiggle`
- `gaussian_noise`
- `color_jiggle_and_gaussian_noise`


## Notes for Reproducibility

- The spawn parameters for Town04 in settings.py are manually configured for the current project task. It is not recommended to migrate them directly to other maps without modification.
- The project relies on CARLA_ROOT to automatically start the simulator. Training cannot start properly if this environment variable is not set.
- If a different CARLA version is used, vehicle spawn positions, camera parameters, and runtime stability may need to be adjusted.
- The replay buffer may consume a large amount of memory depending on image size, buffer capacity, and frame stacking. Adjust --replay_buffer_capacity according to your machine configuration.

