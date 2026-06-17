import numpy as np
import torch
import argparse
import os
import math
import time
import json
from datetime import datetime
import psutil
from collections import deque

import utils
from carla_augmentations import make_augmentor
from logger import Logger
from video import VideoRecorder
from cl_cpc import CurlSacAgent, TD3Agent, DDPGAgent
import carla_env


def parse_args():
    parser = argparse.ArgumentParser()

    # Carla environment settings
    parser.add_argument('--carla_town', default='Town04', type=str)
    parser.add_argument('--max_npc_vehicles', default=10, type=int)
    parser.add_argument('--desired_speed', default=63, type=int)  # km/h (kept for compatibility)
    parser.add_argument('--max_stall_time', default=20, type=int)  # seconds
    parser.add_argument('--stall_speed', default=0.5, type=float)  # km/h
    parser.add_argument('--seconds_per_episode', default=50, type=int)  # seconds
    parser.add_argument('--fps', default=20, type=int)  # Hz
    parser.add_argument('--start_acc_time', default=2.5, type=float)  # seconds (warmup)
    parser.add_argument('--env_verbose', default=False, action='store_true')
    parser.add_argument('--server_port', default=4000, type=int)
    parser.add_argument('--tm_port', default=8000, type=int)

    # ✅ Headway / safety reporting thresholds (for logging & TB)
    parser.add_argument('--thw_good_lo', default=1.5, type=float)
    parser.add_argument('--thw_good_hi', default=2.5, type=float)
    parser.add_argument('--thw_danger', default=1.0, type=float)  # P(THW<thw_danger)

    # Carla camera settings
    parser.add_argument('--camera_image_height', default=90, type=int)
    parser.add_argument('--camera_image_width', default=160, type=int)
    parser.add_argument('--cam_x', default=1.3, type=float)
    parser.add_argument('--cam_y', default=0.0, type=float)
    parser.add_argument('--cam_z', default=1.75, type=float)
    parser.add_argument('--fov', default=110, type=int)
    parser.add_argument('--cam_pitch', default=-15, type=int)

    # Carla reward function parameters/weights (kept for compatibility)
    parser.add_argument('--lambda_r1', default=1.0, type=float)
    parser.add_argument('--lambda_r2', default=0.3, type=float)
    parser.add_argument('--lambda_r3', default=1.0, type=float)
    parser.add_argument('--lambda_r4', default=0.005, type=float)
    parser.add_argument('--lambda_r5', default=1.0, type=float)

    # Image augmentation settings
    parser.add_argument('--augmentation', default='color_jiggle_and_gaussian_noise', type=str)
    parser.add_argument('--frame_stack', default=3, type=int)

    # Replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100_000, type=int)

    # Train
    parser.add_argument('--agent', default='cl_cpc', type=str)
    parser.add_argument('--pixel_sac', default=False, action='store_true')
    parser.add_argument('--init_steps', default=5_000, type=int)
    parser.add_argument('--num_train_steps', default=750_000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)

    # Eval
    parser.add_argument('--eval_freq', default=25_000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)

    # Encoder
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-4, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--predictive_cpc', dest='predictive_cpc', action='store_true')
    parser.add_argument('--no_predictive_cpc', dest='predictive_cpc', action='store_false')
    parser.set_defaults(predictive_cpc=True)

    # Actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)

    # Critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)

    # SAC
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)

    # Misc
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--work_dir_name', default='experiments', type=str)
    parser.add_argument('--save_tb', default=True, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=True, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_freq', default=50000, type=int)
    parser.add_argument('--log_interval', default=500, type=int)
    parser.add_argument('--log_param_hist_imgs', default=False, action='store_true')

    # Debug / stability
    parser.add_argument('--cuda_blocking_debug', default=False, action='store_true',
                        help='Set CUDA_LAUNCH_BLOCKING=1 to get more accurate stack traces.')
    parser.add_argument('--skip_update_on_error', default=True, action='store_true',
                        help='If update() throws, skip this update instead of crashing the whole run.')

    args = parser.parse_args()
    return args


# =========================================================
# Helpers
# =========================================================
def _unwrap_env(env):
    base = env
    while hasattr(base, 'env'):
        base = base.env
    return base


def set_env_eval_flag(env, is_eval: bool):
    base = _unwrap_env(env)
    try:
        base._is_eval = bool(is_eval)
    except Exception:
        pass


def _safe_float(x, default=float('nan')):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _maybe_cuda_sync():
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _get_env_warmup_steps(env, args):
    base = _unwrap_env(env)
    try:
        ws = int(getattr(base, "warmup_steps"))
        if ws > 0:
            return ws
    except Exception:
        pass
    try:
        return int(float(args.start_acc_time) * int(args.fps))
    except Exception:
        return int(2.5 * int(args.fps))


def _get_env_warmup_action(env):
    base = _unwrap_env(env)
    try:
        a = getattr(base, "warmup_action", None)
        if a is not None:
            a = np.array(a, dtype=np.float32).copy()
            if a.shape == (2,):
                return a
    except Exception:
        pass
    return np.array([0.5, 0.0], dtype=np.float32)


def _extract_episode_metrics_from_info(info: dict):
    if info is None:
        info = {}
    out = {}

    for k in ['r1', 'r2', 'r3', 'r4', 'r5', 'r_lane', 'r_center']:
        out[k] = _safe_float(info.get(k, float('nan')))

    # env debug
    out['mean_kmh'] = _safe_float(info.get('mean_kmh', float('nan')))
    out['max_kmh'] = _safe_float(info.get('max_kmh', float('nan')))
    out['brake_sum'] = _safe_float(info.get('brake_sum', float('nan')))
    out['FiT'] = _safe_int(info.get('FiT', 0))

    out['min_TTC'] = _safe_float(info.get('min_TTC', float('nan')))
    out['TTC5'] = _safe_float(info.get('TTC5', float('nan')))
    out['valid_cf_steps'] = _safe_int(info.get('valid_cf_steps', 0))
    out['valid_cf_ratio'] = _safe_float(info.get('valid_cf_ratio', float('nan')))

    out['MAE_dv'] = _safe_float(info.get('MAE_dv', float('nan')))
    out['THW_mean'] = _safe_float(info.get('THW_mean', float('nan')))
    out['THW_p50'] = _safe_float(info.get('THW_p50', float('nan')))
    out['THW_p95'] = _safe_float(info.get('THW_p95', float('nan')))
    out['THW_valid_steps'] = _safe_int(info.get('THW_valid_steps', 0))
    out['THW_valid_ratio'] = _safe_float(info.get('THW_valid_ratio', float('nan')))

    out['same_lane_ratio'] = _safe_float(info.get('same_lane_ratio', float('nan')))
    out['junction_uncertain_ratio'] = _safe_float(info.get('junction_uncertain_ratio', float('nan')))
    out['offlane_streak_max'] = _safe_int(info.get('offlane_streak_max', 0))
    out['leader_missing_streak_max'] = _safe_int(info.get('leader_missing_streak_max', 0))
    out['leader_kick_count'] = _safe_int(info.get('leader_kick_count', 0))

    return out


def _sample_biased_forward_action(env, action_space):
    """
    Prefer using env.sample_action_forward() if available.
    Otherwise fallback to a conservative forward-biased sampler.
    """
    base = _unwrap_env(env)
    if hasattr(base, "sample_action_forward"):
        try:
            a = base.sample_action_forward()
            a = np.array(a, dtype=np.float32).reshape(2,)
            # safety clip
            low = np.array(action_space.low, dtype=np.float32)
            high = np.array(action_space.high, dtype=np.float32)
            return np.clip(a, low, high).astype(np.float32)
        except Exception:
            pass

    low = np.array(action_space.low, dtype=np.float32)
    high = np.array(action_space.high, dtype=np.float32)

    max_tb = float(high[0])
    max_steer = float(high[1])

    p_forward = 0.40  #0.7-->0.40
    r = np.random.rand()
    if r < p_forward:  
        a0 = np.random.uniform(0.10 * max_tb, 0.35 * max_tb) #0.20,0.60-->0.10,0.35
    else:
        a0 = np.random.uniform(-0.90 * max_tb, 0.05 * max_tb) #-0.60,0.10-->-0.90,0.05

    a1 = np.random.normal(loc=0.0, scale=0.05 * max_steer)
    a1 = float(np.clip(a1, -max_steer, max_steer))

    a = np.array([a0, a1], dtype=np.float32)
    a = np.clip(a, low, high).astype(np.float32)
    return a


class EpisodeCFStats:
    """
    Per-episode CF stats computed from step-level info to answer:
      "in safe TTC, make THW mostly in [1.5,2.5]"

    Warmup excluded using step_idx <= warmup_steps
    Leader validity uses info['step_leader_exists']
    """
    def __init__(self, warmup_steps: int, ttc_safe: float, ttc_danger: float,
                 thw_lo: float, thw_hi: float, thw_danger: float):
        self.warmup_steps = int(max(0, warmup_steps))
        self.ttc_safe = float(ttc_safe)
        self.ttc_danger = float(ttc_danger)
        self.thw_lo = float(thw_lo)
        self.thw_hi = float(thw_hi)
        self.thw_danger = float(thw_danger)

        self.cf_steps = 0
        self.ttc_safe_steps = 0
        self.ttc_warn_steps = 0
        self.ttc_danger_steps = 0

        self.thw_valid_steps = 0
        self.thw_in_range_steps = 0
        self.thw_lt_danger_steps = 0

        self.safe_thw_valid_steps = 0
        self.safe_thw_in_range_steps = 0

        self._thw_all = []
        self._thw_safe = []

        # optional: reward step components
        self._step_r_thw = []
        self._step_r_dv = []
        self._step_r_ttc = []

    def update(self, step_idx: int, info: dict):
        if info is None or not isinstance(info, dict):
            return
        if int(step_idx) <= int(self.warmup_steps):
            return

        leader_exists = bool(info.get("step_leader_exists", False))
        if not leader_exists:
            return

        self.cf_steps += 1

        ttc = info.get("step_ttc", float("nan"))
        thw = info.get("step_thw", float("nan"))

        # TTC bucket
        if np.isfinite(ttc):
            t = float(ttc)
            if t > self.ttc_safe:
                self.ttc_safe_steps += 1
                t_bucket = "safe"
            elif t <= self.ttc_danger:
                self.ttc_danger_steps += 1
                t_bucket = "danger"
            else:
                self.ttc_warn_steps += 1
                t_bucket = "warn"
        else:
            # treat nan as warn-ish (not safe)
            self.ttc_warn_steps += 1
            t_bucket = "warn"

        # THW stats
        if np.isfinite(thw):
            h = float(thw)
            self.thw_valid_steps += 1
            self._thw_all.append(h)

            if h < self.thw_danger:
                self.thw_lt_danger_steps += 1
            if (h >= self.thw_lo) and (h <= self.thw_hi):
                self.thw_in_range_steps += 1

            if t_bucket == "safe":
                self.safe_thw_valid_steps += 1
                self._thw_safe.append(h)
                if (h >= self.thw_lo) and (h <= self.thw_hi):
                    self.safe_thw_in_range_steps += 1

        # step reward components (if present)
        sr_thw = info.get("step_r_thw", float("nan"))
        sr_dv = info.get("step_r_dv", float("nan"))
        sr_ttc = info.get("step_r_ttc", float("nan"))
        if np.isfinite(sr_thw):
            self._step_r_thw.append(float(sr_thw))
        if np.isfinite(sr_dv):
            self._step_r_dv.append(float(sr_dv))
        if np.isfinite(sr_ttc):
            self._step_r_ttc.append(float(sr_ttc))

    def finalize(self):
        def _ratio(a, b):
            if b <= 0:
                return float("nan")
            return float(a) / float(b)

        out = {}
        out["cf_steps"] = int(self.cf_steps)

        out["p_ttc_safe"] = _ratio(self.ttc_safe_steps, self.cf_steps)
        out["p_ttc_warn"] = _ratio(self.ttc_warn_steps, self.cf_steps)
        out["p_ttc_danger"] = _ratio(self.ttc_danger_steps, self.cf_steps)

        out["thw_valid_steps_calc"] = int(self.thw_valid_steps)
        out["p_thw_in_1p5_2p5"] = _ratio(self.thw_in_range_steps, self.thw_valid_steps)
        out["p_thw_lt_1p0"] = _ratio(self.thw_lt_danger_steps, self.thw_valid_steps)

        out["safe_thw_valid_steps_calc"] = int(self.safe_thw_valid_steps)
        out["p_safe_thw_in_1p5_2p5"] = _ratio(self.safe_thw_in_range_steps, self.safe_thw_valid_steps)

        # distribution (all)
        if len(self._thw_all) > 0:
            arr = np.array(self._thw_all, dtype=np.float32)
            out["thw_p05_calc"] = float(np.percentile(arr, 5))
            out["thw_std_calc"] = float(np.std(arr))
            out["thw_iqr_calc"] = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        else:
            out["thw_p05_calc"] = float("nan")
            out["thw_std_calc"] = float("nan")
            out["thw_iqr_calc"] = float("nan")

        # distribution (safe subset)
        if len(self._thw_safe) > 0:
            arrs = np.array(self._thw_safe, dtype=np.float32)
            out["safe_thw_mean_calc"] = float(np.mean(arrs))
            out["safe_thw_p50_calc"] = float(np.percentile(arrs, 50))
            out["safe_thw_p95_calc"] = float(np.percentile(arrs, 95))
        else:
            out["safe_thw_mean_calc"] = float("nan")
            out["safe_thw_p50_calc"] = float("nan")
            out["safe_thw_p95_calc"] = float("nan")

        # step reward components mean
        out["step_r_thw_mean"] = float(np.mean(self._step_r_thw)) if len(self._step_r_thw) > 0 else float("nan")
        out["step_r_dv_mean"] = float(np.mean(self._step_r_dv)) if len(self._step_r_dv) > 0 else float("nan")
        out["step_r_ttc_mean"] = float(np.mean(self._step_r_ttc)) if len(self._step_r_ttc) > 0 else float("nan")

        return out


def run_eval_loop(env, agent, augmentor, video, num_episodes, L, step, args, video_dir,
                  sample_stochastically=False):
    set_env_eval_flag(env, True)

    try:
        all_ep_rewards = []
        all_ep_steps = []
        all_ep_infos = {}

        best_episode = {'reward': -math.inf, 'ep': -1}

        base_env = _unwrap_env(env)
        warmup_steps = _get_env_warmup_steps(env, args)
        warmup_action = _get_env_warmup_action(env)

        # thresholds from env
        ttc_safe = float(getattr(base_env, "ttc_safe", 4.0))
        ttc_danger = float(getattr(base_env, "ttc_danger", 2.0))

        for i in range(num_episodes):
            obs = env.reset()

            stats = EpisodeCFStats(
                warmup_steps=warmup_steps,
                ttc_safe=ttc_safe,
                ttc_danger=ttc_danger,
                thw_lo=float(args.thw_good_lo),
                thw_hi=float(args.thw_good_hi),
                thw_danger=float(args.thw_danger),
            )

            video.init(enabled=True)
            done = False
            info = None
            episode_reward = 0.0
            episode_steps = 0

            while not done:
                time.sleep(1.0 / float(args.fps))
                obs_aug = augmentor.evaluation_augmentation(obs)

                if episode_steps < warmup_steps:
                    action = warmup_action.copy()
                else:
                    with utils.eval_mode(agent):
                        env.curl_driving = True
                        if sample_stochastically:
                            action = agent.sample_action(obs_aug)
                        else:
                            action = agent.select_action(obs_aug)

                obs, reward, done, info = env.step(action)
                episode_steps += 1

                # step stats update (use step_idx = episode_steps)
                stats.update(step_idx=episode_steps, info=info)

                video.record(env)
                episode_reward += float(reward)

            if episode_reward > best_episode['reward']:
                best_episode['reward'] = episode_reward
                best_episode['ep'] = i

            video.save(f'eval_step_{step}_ep_{i+1}.mp4')
            all_ep_rewards.append(episode_reward)
            all_ep_steps.append(episode_steps)

            ep_metrics = _extract_episode_metrics_from_info(info)
            cf_metrics = stats.finalize()

            # merge
            merged = {}
            merged.update(ep_metrics)
            merged.update(cf_metrics)

            for k, v in merged.items():
                all_ep_infos.setdefault(k, []).append(v)

        # keep only best ep video
        if num_episodes > 1 and best_episode['ep'] >= 0:
            for ep in range(num_episodes):
                if ep == best_episode['ep']:
                    continue
                try:
                    p = os.path.join(video_dir, f'eval_step_{step}_ep_{ep+1}.mp4')
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

        # default eval reward stats (console-friendly)
        L.log('eval/max_ep_reward', float(np.max(all_ep_rewards)), step)
        L.log('eval/mean_ep_reward', float(np.mean(all_ep_rewards)), step)
        L.log('eval/min_ep_reward', float(np.min(all_ep_rewards)), step)
        L.log('eval/std_ep_reward', float(np.std(all_ep_rewards)), step)

        L.log('eval/max_ep_steps', float(np.max(all_ep_steps)), step)
        L.log('eval/mean_ep_steps', float(np.mean(all_ep_steps)), step)
        L.log('eval/min_ep_steps', float(np.min(all_ep_steps)), step)
        L.log('eval/std_ep_steps', float(np.std(all_ep_steps)), step)

        if sample_stochastically:
            L.log('eval/stochastic_max_ep_reward', float(np.max(all_ep_rewards)), step)
            L.log('eval/stochastic_mean_ep_reward', float(np.mean(all_ep_rewards)), step)
            L.log('eval/stochastic_min_ep_reward', float(np.min(all_ep_rewards)), step)
            L.log('eval/stochastic_std_ep_reward', float(np.std(all_ep_rewards)), step)

        # richer eval metrics into TB (not printed in eval console line)
        for k, v in all_ep_infos.items():
            if isinstance(v, list) and len(v) > 0:
                try:
                    v_arr = np.array(v, dtype=float)
                    if np.any(np.isfinite(v_arr)):
                        L.log(f'eval/ep_mean/{k}', float(np.nanmean(v_arr)), step)
                        L.log(f'eval/ep_std/{k}', float(np.nanstd(v_arr)), step)
                except Exception:
                    pass

        return L

    finally:
        set_env_eval_flag(env, False)


def make_agent(obs_shape, action_shape, args, device, augmentor):
    if not hasattr(args, 'pixel_sac'):
        args.pixel_sac = False

    if args.agent == 'cl_cpc':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            augmentor=augmentor,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            log_param_hist_imgs=args.log_param_hist_imgs,
            detach_encoder=args.detach_encoder,
            pixel_sac=args.pixel_sac,
            predictive_cpc=args.predictive_cpc,
        )
    elif args.agent == 'td3':
        return TD3Agent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            augmentor=augmentor,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            tau=args.critic_tau,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
        )
    elif args.agent == 'ddpg':
        return DDPGAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            augmentor=augmentor,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            tau=args.critic_tau,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            encoder_feature_dim=args.encoder_feature_dim,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            pixel_sac=args.pixel_sac,
            predictive_cpc=args.predictive_cpc
        )
    else:
        raise AssertionError('agent is not supported: %s' % args.agent)


def make_env(args):
    env = carla_env.CarlaEnv(
        carla_town=args.carla_town,
        max_npc_vehicles=args.max_npc_vehicles,
        desired_speed=args.desired_speed,
        max_stall_time=args.max_stall_time,
        stall_speed=args.stall_speed,
        seconds_per_episode=args.seconds_per_episode,
        fps=args.fps,
        server_port=args.server_port,
        tm_port=args.tm_port,
        verbose=args.env_verbose,
        pre_transform_image_height=args.camera_image_height,
        pre_transform_image_width=args.camera_image_width,
        fov=args.fov,
        cam_x=args.cam_x,
        cam_y=args.cam_y,
        cam_z=args.cam_z,
        cam_pitch=args.cam_pitch,
        lambda_r1=args.lambda_r1,
        lambda_r2=args.lambda_r2,
        lambda_r3=args.lambda_r3,
        lambda_r4=args.lambda_r4,
        lambda_r5=args.lambda_r5,
    )

    # Force env warmup to follow CLI, avoid train/env mismatch
    try:
        env.start_acc_time = float(args.start_acc_time)
        env.warmup_steps = int(float(env.start_acc_time) * int(env.fps))
        env.warmup_action = np.array([0.35, 0.0], dtype=np.float32)  #1：0.5 -->2：0.35
        try:
            env.leader_kick_max_steps = env.warmup_steps + max(1, int(0.5 * env.fps))
        except Exception:
            pass
    except Exception:
        pass

    env.seed(args.seed)
    env.reset()

    env = utils.FrameStack(env, k=args.frame_stack)
    return env


def main():
    args = parse_args()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)

    assert args.save_freq % args.eval_freq == 0, 'Save frequency must be a multiple of eval frequency'

    utils.set_seed_everywhere(args.seed)

    if args.cuda_blocking_debug:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        print("[train.py] CUDA_LAUNCH_BLOCKING=1 enabled for debugging")

    if args.pixel_sac is True:
        print('[train.py] Pixel SAC mode selected, disabling contrastive objectives while keeping the configured image augmentation.')

    camera_image_shape = (args.camera_image_height, args.camera_image_width)
    augmentor = make_augmentor(args.augmentation, camera_image_shape)

    args.augmented_image_height = augmentor.output_shape[0]
    args.augmented_image_width = augmentor.output_shape[1]

    working_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.work_dir_name)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    ts = datetime.now().strftime("%m-%d--%H-%M-%S")
    env_name = args.carla_town
    exp_type = 'pixel_sac' if args.pixel_sac else str(args.augmentation)
    if args.detach_encoder:
        exp_type += '_detached'

    exp_name = (
        env_name + '--' + ts + '--im' + str(args.camera_image_height) + 'x' + str(args.camera_image_width)
        + '-b' + str(args.batch_size) + '-s' + str(args.seed) + '-' + exp_type
    )

    working_dir = os.path.join(working_dir, exp_name)
    utils.make_dir(working_dir)

    video_dir = utils.make_dir(os.path.join(working_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(working_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(working_dir, 'buffer'))

    L = Logger(working_dir, use_tb=args.save_tb)

    env = make_env(args)
    base_env = _unwrap_env(env)

    vid_path = video_dir if args.save_video else None
    video = VideoRecorder(vid_path, base_env.fps)

    with open(os.path.join(working_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4, ensure_ascii=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape
    pre_aug_obs_shape = env.observation_space.shape
    obs_shape = (3 * args.frame_stack, args.augmented_image_height, args.augmented_image_width)

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        augmentor=augmentor
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device,
        augmentor=augmentor
    )

    episode, episode_reward, done, info = 0, 0.0, True, None

    fps = 0.0
    sys_mem_pcnt = 0.0
    proc_mem_GB = 0.0
    sys_mem = deque(maxlen=int(args.log_interval))
    proc_mem = deque(maxlen=int(args.log_interval))

    try:
        max_episode_reward = (base_env.desired_speed / 3.6) * base_env.dt * base_env._max_episode_steps
        print(f'Maximum episode reward possible for requested CarlaEnv configuration: {round(max_episode_reward, 2)}')
    except Exception:
        max_episode_reward = 1.0
        print('Maximum episode reward estimate unavailable; using fallback.')

    def buffer_ready_for_update():
        try:
            if hasattr(replay_buffer, "full") and replay_buffer.full:
                return True
            if hasattr(replay_buffer, "idx"):
                return int(replay_buffer.idx) >= int(args.batch_size)
            if hasattr(replay_buffer, "_idx"):
                return int(replay_buffer._idx) >= int(args.batch_size)
        except Exception:
            pass
        return False

    episode_step = 0
    start_time = time.time()

    # episode CF stats accumulator
    def _new_episode_stats():
        warmup_steps = _get_env_warmup_steps(env, args)
        ttc_safe = float(getattr(base_env, "ttc_safe", 4.0))
        ttc_danger = float(getattr(base_env, "ttc_danger", 2.0))
        return EpisodeCFStats(
            warmup_steps=warmup_steps,
            ttc_safe=ttc_safe,
            ttc_danger=ttc_danger,
            thw_lo=float(args.thw_good_lo),
            thw_hi=float(args.thw_good_hi),
            thw_danger=float(args.thw_danger),
        )

    ep_stats = _new_episode_stats()

    for step in range(args.num_train_steps + 1):

        if step == args.init_steps:
            start_time = time.time()

        # ✅ eval trigger
        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            if getattr(base_env, "verbose", False):
                print('episode done: evaluation starts')
            print(f'[train.py] Started evaluation loop at step {step}')

            if args.num_eval_episodes > 0:
                if step > 0 and step % args.num_train_steps == 0:
                    L = run_eval_loop(env, agent, augmentor, video, 50, L, step, args, video_dir, sample_stochastically=False)
                else:
                    L = run_eval_loop(env, agent, augmentor, video, args.num_eval_episodes, L, step, args, video_dir, sample_stochastically=False)

            done = True
            print(f'[train.py] Finished evaluation loop at step {step}')

            if step % args.save_freq == 0:
                if args.save_model:
                    agent.save(model_dir, args.augmentation, step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                max_score_achieved = episode_reward / max_episode_reward if max_episode_reward != 0 else 0.0

                L.log('train/ep_steps', episode_step, step)
                L.log('train/ep_reward', episode_reward, step)
                L.log('train/ep_max_score_ratio', max_score_achieved, step)

                if step > args.init_steps:
                    L.log('train/ep_mean_fps', fps, step)

                # log per-episode env summary (from last info)
                if info is not None:
                    m = _extract_episode_metrics_from_info(info)

                    # reward sums
                    L.log('train/reward/r1_THW_sum', _safe_float(m.get('r1', 0.0), 0.0), step)
                    L.log('train/reward/r2_dv_sum', _safe_float(m.get('r2', 0.0), 0.0), step)
                    L.log('train/reward/r3_TTCrisk_sum', _safe_float(m.get('r3', 0.0), 0.0), step)
                    L.log('train/reward/r4_collision_sum', _safe_float(m.get('r4', 0.0), 0.0), step)
                    L.log('train/reward/r5_unused_sum', _safe_float(m.get('r5', 0.0), 0.0), step)

                    # speed
                    L.log('train/speed/mean_kmh', _safe_float(m.get('mean_kmh')), step)
                    L.log('train/speed/max_kmh', _safe_float(m.get('max_kmh')), step)
                    L.log('train/speed/brake_sum', _safe_float(m.get('brake_sum')), step)

                    # finish type
                    L.log('train/FiT', _safe_int(m.get('FiT', 0)), step)

                    # safety TTC
                    L.log('train/safety/min_TTC', _safe_float(m.get('min_TTC')), step)
                    L.log('train/safety/TTC5', _safe_float(m.get('TTC5')), step)

                    # validity
                    L.log('train/cf/valid_cf_steps', _safe_int(m.get('valid_cf_steps', 0)), step)
                    L.log('train/cf/valid_cf_ratio', _safe_float(m.get('valid_cf_ratio')), step)

                    # headway / dv
                    L.log('train/stability/MAE_dv', _safe_float(m.get('MAE_dv')), step)

                    L.log('train/headway/THW_mean', _safe_float(m.get('THW_mean')), step)
                    L.log('train/headway/THW_p50', _safe_float(m.get('THW_p50')), step)
                    L.log('train/headway/THW_p95', _safe_float(m.get('THW_p95')), step)

                    L.log('train/headway/THW_valid_steps', _safe_int(m.get('THW_valid_steps', 0)), step)
                    L.log('train/headway/THW_valid_ratio', _safe_float(m.get('THW_valid_ratio')), step)

                    # robustness
                    L.log('train/robust/same_lane_ratio', _safe_float(m.get('same_lane_ratio')), step)
                    L.log('train/robust/junction_uncertain_ratio', _safe_float(m.get('junction_uncertain_ratio')), step)
                    L.log('train/robust/offlane_streak_max', _safe_int(m.get('offlane_streak_max', 0)), step)
                    L.log('train/robust/leader_missing_streak_max', _safe_int(m.get('leader_missing_streak_max', 0)), step)
                    L.log('train/leader/leader_kick_count', _safe_int(m.get('leader_kick_count', 0)), step)

                # ✅ log per-episode derived CF stats (core goal)
                cf = ep_stats.finalize()
                L.log('train/cf/cf_steps_calc', _safe_int(cf.get('cf_steps', 0)), step)

                L.log('train/safety/p_ttc_safe', _safe_float(cf.get('p_ttc_safe', float('nan'))), step)
                L.log('train/safety/p_ttc_warn', _safe_float(cf.get('p_ttc_warn', float('nan'))), step)
                L.log('train/safety/p_ttc_danger', _safe_float(cf.get('p_ttc_danger', float('nan'))), step)

                L.log('train/headway/p_in_1p5_2p5', _safe_float(cf.get('p_thw_in_1p5_2p5', float('nan'))), step)
                L.log('train/headway/p_safe_in_1p5_2p5', _safe_float(cf.get('p_safe_thw_in_1p5_2p5', float('nan'))), step)
                L.log('train/headway/p_lt_1p0', _safe_float(cf.get('p_thw_lt_1p0', float('nan'))), step)

                L.log('train/headway/thw_p05_calc', _safe_float(cf.get('thw_p05_calc', float('nan'))), step)
                L.log('train/headway/thw_std_calc', _safe_float(cf.get('thw_std_calc', float('nan'))), step)
                L.log('train/headway/thw_iqr_calc', _safe_float(cf.get('thw_iqr_calc', float('nan'))), step)

                L.log('train/headway/safe_thw_mean_calc', _safe_float(cf.get('safe_thw_mean_calc', float('nan'))), step)
                L.log('train/headway/safe_thw_p50_calc', _safe_float(cf.get('safe_thw_p50_calc', float('nan'))), step)
                L.log('train/headway/safe_thw_p95_calc', _safe_float(cf.get('safe_thw_p95_calc', float('nan'))), step)

                # step reward components mean (debug if reward shaping behaves)
                L.log('train/reward/step_r_thw_mean', _safe_float(cf.get('step_r_thw_mean', float('nan'))), step)
                L.log('train/reward/step_r_dv_mean', _safe_float(cf.get('step_r_dv_mean', float('nan'))), step)
                L.log('train/reward/step_r_ttc_mean', _safe_float(cf.get('step_r_ttc_mean', float('nan'))), step)

            L.dump(step)

            # reset episode
            start_time = time.time()
            obs = env.reset()
            done = False
            episode_reward = 0.0
            episode_step = 0
            episode += 1
            L.log('train/episode', episode, step)

            # new episode stats accumulator
            ep_stats = _new_episode_stats()

        warmup_steps = _get_env_warmup_steps(env, args)
        warmup_action = _get_env_warmup_action(env)

        if episode_step < warmup_steps:
            action = warmup_action.copy()
        else:
            if step < args.init_steps:
                action = _sample_biased_forward_action(env, env.action_space)
            else:
                with utils.eval_mode(agent):
                    env.curl_driving = True
                    action = agent.sample_action(obs)

        next_obs, reward, done, info = env.step(action)

        # update per-step CF stats (use step_idx = episode_step+1)
        try:
            ep_stats.update(step_idx=int(episode_step + 1), info=info)
        except Exception:
            pass

        # memory tracking
        sys_mem.append(psutil.virtual_memory().percent)
        proc_mem.append(round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3), 4))

        try:
            done_bool = 0.0 if (episode_step + 1 == int(env._max_episode_steps)) else float(done)
        except Exception:
            done_bool = float(done)

        episode_reward += float(reward)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        obs = next_obs
        episode_step += 1

        if step >= args.init_steps and buffer_ready_for_update():
            try:
                if episode_step < warmup_steps:
                    agent.update(replay_buffer, L, step, only_cpc=True)
                else:
                    agent.update(replay_buffer, L, step)
            except Exception as e:
                _maybe_cuda_sync()
                print(f"\n[train.py][WARNING] agent.update() failed at step={step}, ep_step={episode_step}.")
                print(f"[train.py][WARNING] Exception: {repr(e)}")
                if not args.skip_update_on_error:
                    raise

        if step >= args.init_steps:
            try:
                fps = round(episode_step / (time.time() - start_time), 2)
            except Exception:
                fps = 0.0

        sys_mem_pcnt = round(sum(sys_mem) / len(sys_mem), 2) if len(sys_mem) > 0 else 0.0
        proc_mem_GB = round(sum(proc_mem) / len(proc_mem), 4) if len(proc_mem) > 0 else 0.0

        if step % args.log_interval == 0:
            L.log('train/system/mean_sys_mem_pcnt', sys_mem_pcnt, step)
            L.log('train/system/mean_proc_mem_GB', proc_mem_GB, step)

            # light-weight step debug into TB (NOT console)
            if info is not None and isinstance(info, dict):
                L.log('train/step/ttc', _safe_float(info.get('step_ttc', float('nan'))), step)
                L.log('train/step/thw', _safe_float(info.get('step_thw', float('nan'))), step)
                L.log('train/step/gap', _safe_float(info.get('step_gap', float('nan'))), step)
                L.log('train/step/dv', _safe_float(info.get('step_dv', float('nan'))), step)
                L.log('train/step/leader_exists', 1.0 if bool(info.get('step_leader_exists', False)) else 0.0, step)

    env.deactivate()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

    print('-' * 50)
    print('PyTorch version:', torch.__version__)
    print('CUDA availability:', torch.cuda.is_available())
    print('CUDA device count:', torch.cuda.device_count())
    print('CUDA current device:', torch.cuda.current_device())
    print('CUDA device name (0):', torch.cuda.get_device_name(0))
    print('-' * 50)

    main()
