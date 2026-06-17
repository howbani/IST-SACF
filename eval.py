import os
import argparse
import csv
import torch
import numpy as np
import json
import carla

import utils
from torch.utils.tensorboard import SummaryWriter
from carla_augmentations import make_augmentor
import carla_env
from video import VideoRecorder
from train import make_agent

USE_NOVEL_PRESETS = True   #为True时表示评估使用“新颖/未在训练中出现”的天气预设

if USE_NOVEL_PRESETS:
    # Evaluation weather presets
    WEATHER_PRESETS =  {'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
                        'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
                        'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
                        'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
                        'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
                        'HardRainNoon': carla.WeatherParameters.HardRainNoon,
                        'HardRainSunset': carla.WeatherParameters.HardRainSunset}

else:
    # Training weather presets
    WEATHER_PRESETS = {'ClearNoon': carla.WeatherParameters.ClearNoon,
                        'ClearSunset': carla.WeatherParameters.ClearSunset,
                        'CloudyNoon': carla.WeatherParameters.CloudyNoon,
                        'CloudySunset': carla.WeatherParameters.CloudySunset,
                        'WetNoon': carla.WeatherParameters.WetNoon,
                        'WetSunset': carla.WeatherParameters.WetSunset,
                        'MidRainSunset': carla.WeatherParameters.MidRainSunset}
                    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir_path', default='', type=str)
    parser.add_argument('--model_step', default=1_000_000, type=int)  #添加整数参数 --model_step，指定要加载的模型训练步数
    parser.add_argument('--env_verbose', default=False, action='store_true')  #日志
    args = parser.parse_args()
    return args


def run_eval_loop(env, agent, augmentor, step, experiment_dir_path, num_episodes=10, record_video=False):
        #参数包括环境、智能体、增强器、模型步数、实验目录、评估幕数（默认10，主程序中实际是50）以及是否录制视频的开关
        # Initializations
        exp_name = os.path.basename(experiment_dir_path)  #取实验目录的最后一级名称作为实验名基础
        exp_name = exp_name.split('-')[-1]
        print(f'Running evaluation loop for experiment {exp_name}')
        ep_rewards = []   #初始化每幕累计奖励的列表，用于后续统计
        ep_steps = []     #初始化每幕步数的列表
        path = os.path.join(experiment_dir_path, 'eval_videos')  #实验目录下拼接出视频输出目录 eval_videos
        tb_path = os.path.join(experiment_dir_path, 'eval_tb_logs', f'model_{step}')
        writer = SummaryWriter(log_dir=tb_path)
        total_eval_steps = 0
        if record_video:  #检查视频目录是否存在
            if not os.path.exists(path):  #目录不存在
                os.mkdir(path)  #创建目录
            else:  #如果目录已存在，清空其中旧文件，避免新评估与旧视频混淆
                for file in os.listdir(path):
                    os.remove(os.path.join(path, file))
        video = VideoRecorder(path, env.fps)   #创建视频录制器

        # Setup detailed logging
        detailed_log_dir = os.path.join(experiment_dir_path, 'eval_detailed_logs', f'model_{step}')
        if not os.path.exists(detailed_log_dir):
            os.makedirs(detailed_log_dir)
        
        f_vel = open(os.path.join(detailed_log_dir, 'velocity.csv'), 'w', newline='')
        f_ttc = open(os.path.join(detailed_log_dir, 'ttc.csv'), 'w', newline='')
        f_thw = open(os.path.join(detailed_log_dir, 'thw.csv'), 'w', newline='')
        f_smooth = open(os.path.join(detailed_log_dir, 'smoothness.csv'), 'w', newline='')
        
        writer_vel = csv.writer(f_vel)
        writer_ttc = csv.writer(f_ttc)
        writer_thw = csv.writer(f_thw)
        writer_smooth = csv.writer(f_smooth)
        
        writer_vel.writerow(['global_step', 'episode', 'step', 'v_ego_kmh', 'v_lead_kmh'])
        writer_ttc.writerow(['global_step', 'episode', 'step', 'ttc'])
        writer_thw.writerow(['global_step', 'episode', 'step', 'thw'])
        writer_smooth.writerow(['global_step', 'episode', 'step', 'acc_ms2', 'jerk_ms3'])

        # Run evaluation loop  执行评估循环
        for i in range(num_episodes):
            obs = env.reset()
            chosen_preset = list(WEATHER_PRESETS.keys())[env.weather_preset_idx]
            video.init(enabled=record_video)  #初始化本幕的视频录制；只有当record_video=True时才实际启用
            done = False  #设置幕结束标志为未结束
            episode_reward = 0  #累计奖励清零
            episode_step = 0    #步数计数清零
            # statistics for this episode
            last_info = {}
            thw_valid_steps = 0
            thw_in_1p5_2p5_steps = 0
            thw_in_1p0_4p0_steps = 0
            ttc_valid_steps = 0
            ttc_danger_steps = 0
            
            # New metrics collection
            episode_vel_diffs = []
            episode_thws = []
            
            # Smoothness state
            prev_v_ms = 0.0
            prev_acc_ms2 = 0.0

            while not done:

                # Perform anchor augmentation
                obs = augmentor.evaluation_augmentation(obs)  #对观测执行评估用的数据增强（图像处理）

                # Sample action from agent
                with utils.eval_mode(agent):  #切换智能体到评估模式的上下文
                    action = agent.sample_action(obs)  #让智能体基于当前观测产生一个动作；在评估模式下一般是确定性的或低噪声策略

                # Take step in environment
                obs, reward, done, info = env.step(action)  #动作送入环境，获得下一观测、即时奖励、是否结束标志与附加信息

                v_ego_kmh = info.get('step_v_ego_ms', 0.0) * 3.6
                v_lead_raw = info.get('step_v_lead', None)
                v_lead_kmh = v_lead_raw * 3.6 if v_lead_raw is not None else None
                gap = info.get('step_gap', 0.0)
                thw = info.get('step_thw', 0.0)
                ttc = info.get('step_ttc', 100.0)
                fit = info.get('FiT', 0)

                # accumulate per-step THW/TTC stats (skip invalid values)
                if thw is not None and np.isfinite(thw):
                    thw_valid_steps += 1
                    if 1.5 <= thw <= 2.5:
                        thw_in_1p5_2p5_steps += 1
                    if 1.0 <= thw <= 4.0:
                        thw_in_1p0_4p0_steps += 1

                if ttc is not None and np.isfinite(ttc):
                    ttc_valid_steps += 1
                    if ttc < 2.51:
                        ttc_danger_steps += 1

                # Collect data for new metrics
                if v_lead_kmh is not None:
                    episode_vel_diffs.append(abs(v_ego_kmh - v_lead_kmh))
                
                if thw is not None and np.isfinite(thw):
                    episode_thws.append(thw)

                last_info = info

                writer.add_scalar('Eval_Step/Velocity_Ego_kmh', v_ego_kmh, total_eval_steps)
                if v_lead_kmh is not None:
                    writer.add_scalar('Eval_Step/Velocity_Leader_kmh', v_lead_kmh, total_eval_steps)
                writer.add_scalar('Eval_Step/Gap_m', gap, total_eval_steps)
                writer.add_scalar('Eval_Step/THW_s', thw, total_eval_steps)
                writer.add_scalar('Eval_Step/TTC_s', ttc, total_eval_steps)
                total_eval_steps += 1

                # Calculate acc and jerk
                dt = 1.0 / env.fps
                current_v_ms = info.get('step_v_ego_ms', 0.0)
                
                if episode_step == 0:
                     acc_ms2 = 0.0
                     jerk_ms3 = 0.0
                else:
                     acc_ms2 = (current_v_ms - prev_v_ms) / dt
                     jerk_ms3 = (acc_ms2 - prev_acc_ms2) / dt
                
                prev_v_ms = current_v_ms
                prev_acc_ms2 = acc_ms2

                # Write detailed logs
                writer_vel.writerow([total_eval_steps, i+1, episode_step, v_ego_kmh, v_lead_kmh if v_lead_kmh is not None else ''])
                writer_ttc.writerow([total_eval_steps, i+1, episode_step, ttc])
                writer_thw.writerow([total_eval_steps, i+1, episode_step, thw])
                writer_smooth.writerow([total_eval_steps, i+1, episode_step, acc_ms2, jerk_ms3])

                # Administration and logging
                video.record(env)
                episode_reward += reward  #累加本幕的总奖励
                episode_step += 1
                    
            video.save(f'{step}_{i+1}_r{int(episode_reward)}.mp4')
            ep_steps.append(episode_step)  #记录本幕的总步数
            ep_rewards.append(episode_reward)  #记录本幕的累计奖励
            writer.add_scalar('Eval_Episode/Reward', episode_reward, i)
            writer.add_scalar('Eval_Episode/Length', episode_step, i)

            # compute episode-level ratios (use valid steps as denominator)
            p_thw_1p5_2p5 = float(thw_in_1p5_2p5_steps) / thw_valid_steps if thw_valid_steps > 0 else float('nan')
            p_thw_1p0_4p0 = float(thw_in_1p0_4p0_steps) / thw_valid_steps if thw_valid_steps > 0 else float('nan')
            p_ttc_danger = float(ttc_danger_steps) / ttc_valid_steps if ttc_valid_steps > 0 else float('nan')

            # FiT: use last info's FiT if available
            fit_final = int(last_info.get('FiT', fit)) if isinstance(last_info, dict) else int(fit)

            # Compute new metrics
            mean_vel_diff_mae = np.mean(episode_vel_diffs) if len(episode_vel_diffs) > 0 else float('nan')
            mean_thw = np.mean(episode_thws) if len(episode_thws) > 0 else float('nan')
            # THW RMSE with target 2.0s
            if len(episode_thws) > 0:
                thw_rmse = np.sqrt(np.mean((np.array(episode_thws) - 2.0)**2))
            else:
                thw_rmse = float('nan')

            # log to tensorboard
            writer.add_scalar('Eval_Episode/p_thw_1p5_2p5', p_thw_1p5_2p5, i)
            writer.add_scalar('Eval_Episode/p_thw_1p0_4p0', p_thw_1p0_4p0, i)
            writer.add_scalar('Eval_Episode/p_ttc_lt_2p51', p_ttc_danger, i)
            writer.add_scalar('Eval_Episode/VelDiffMAE', mean_vel_diff_mae, i)
            writer.add_scalar('Eval_Episode/MeanTHW', mean_thw, i)
            writer.add_scalar('Eval_Episode/THW_RMSE', thw_rmse, i)

            print('Episode %d/%d | Weather preset: %s | Cumulative reward: %f | Steps: %f | FiT: %d | pTHW[1.5,2.5]=%.3f | pTHW[1.0,4.0]=%.3f | pTTC<2.51=%.3f | VelDiffMAE=%.3f | MeanTHW=%.3f | THW_RMSE=%.3f' % (
                i + 1, num_episodes, chosen_preset, episode_reward, episode_step, fit_final, p_thw_1p5_2p5, p_thw_1p0_4p0, p_ttc_danger, mean_vel_diff_mae, mean_thw, thw_rmse))

        writer.close()

        # Close detailed logs
        f_vel.close()
        f_ttc.close()
        f_thw.close()
        f_smooth.close()

        return ep_rewards, ep_steps

def make_env(args, weather_presets):

    # Initialize the CARLA environment
    env = carla_env.CarlaEnv(args.carla_town, args.max_npc_vehicles, 
                   args.desired_speed, args.max_stall_time, args.stall_speed, args.seconds_per_episode,
                   args.fps, 4000, 8000, args.env_verbose, args.camera_image_height, args.camera_image_width, 
                   args.fov, args.cam_x, args.cam_y, args.cam_z, args.cam_pitch,
                   args.lambda_r1, args.lambda_r2, args.lambda_r3, args.lambda_r4, args.lambda_r5,
                   weather_presets=weather_presets)
    
    # Set the random seed and reset
    env.seed(args.seed)
    env.reset()

    # Wrap CarlaEnv in FrameStack class to stack several consecutive frames together
    env = utils.FrameStack(env, k=args.frame_stack)

    return env

def main():

    # Parse arguments
    args = parse_args()
    verbose = False
    if args.env_verbose:
        verbose = True
    with open(os.path.join(args.experiment_dir_path, 'args.json'), 'r') as f:
        args.__dict__.update(json.load(f))
    if verbose: args.env_verbose = True

    # Set a fixed random seed for fair comparison across experiments.
    # Use seed 42 for evaluation runs to reproduce results.
    args.seed = 42

    # Random seed
    utils.set_seed_everywhere(args.seed)

    # Anchor/target data augmentor
    camera_image_shape = (args.camera_image_height, args.camera_image_width)
    augmentor = make_augmentor(args.augmentation, camera_image_shape)

    # In the evaluation, only novel weather presets are used in order 
    # to test the generalization/robustness capabilities of the agent.a
    env = make_env(args, list(WEATHER_PRESETS.values()))

    # Shapes
    action_shape = env.action_space.shape
    pre_aug_obs_shape = env.observation_space.shape
    obs_shape = (3*args.frame_stack, args.augmented_image_height, args.augmented_image_width)

    # Make use of GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up agent
    agent = make_agent(obs_shape, action_shape, args, device, augmentor)

    # Load model
    model_dir_path = os.path.join(args.experiment_dir_path, 'model')
    agent.load(model_dir_path, str(args.augmentation), str(args.model_step))

    # Run evaluation loop
    ep_rewards, ep_steps = run_eval_loop(env, agent, augmentor, args.model_step, args.experiment_dir_path, num_episodes=50, record_video=False)

    # Deactivate the environment
    env.deactivate()

    # Print results
    print()
    print('Average reward: %f' % np.mean(ep_rewards))
    print('Max reward: %f' % np.max(ep_rewards))
    print('Min reward: %f' % np.min(ep_rewards))
    print('Std reward: %f' % np.std(ep_rewards))
    print()
    print('Average steps: %f' % np.mean(ep_steps))
    print('Max steps: %f' % np.max(ep_steps))
    print('Min steps: %f' % np.min(ep_steps))
    print('Std steps: %f' % np.std(ep_steps))

if __name__ == "__main__":
    main()
