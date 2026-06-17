# carla_env.py (FULL COPY-PASTE)
#
# Pure longitudinal car-following (ego longitudinal RL + env lateral autopilot)
# - Lane invasion detection REMOVED completely
# - Robust same_lane/valid_cf via route-projection (junction-robust)
# - Reward: explicitly drives THW into [1.5, 2.5] *under safe TTC*
# - Warm-up (first 2.5s): fixed action, reward=0, excluded from CF metrics
#
# FinishType (FiT):
# 0=running/none, 1=collision, 2=time_up, 3=stall

# Standard library
import os
import random
import time
import math
import queue
import shutil
import pkg_resources

# Installed
import carla
import numpy as np
import cv2
import gymnasium as gym

gym.logger.set_level(40)

# Modules
import settings
from carla_handler import CarlaServer

# ----------------------
# Constants
# ----------------------
TIMEOUT = 30.0
RENDER_WIDTH = 1152
RENDER_HEIGHT = 640
G = 9.807
MAX_TTC = 60.0

CARLA_VERSION_STR = pkg_resources.get_distribution("carla").version


def _parse_carla_version_tuple(ver_str: str):
    try:
        parts = str(ver_str).split(".")
        return tuple(int(x) for x in parts[:3])
    except Exception:
        return (0, 0, 0)


CARLA_VERSION_TUPLE = _parse_carla_version_tuple(CARLA_VERSION_STR)

# 0.9.14+ spawn Z is more sensitive
SPAWN_HEIGHT = 0.5 if CARLA_VERSION_TUPLE >= (0, 9, 14) else 0.2


# ============================================================
# Simple PID helpers
# ============================================================
class PID:
    def __init__(self, kp, ki, kd, dt, integrator_limit=1.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.integrator_limit = float(integrator_limit)
        self._i = 0.0
        self._prev_e = None

    def reset(self):
        self._i = 0.0
        self._prev_e = None

    def step(self, e):
        e = float(e)
        self._i += e * self.dt
        self._i = float(np.clip(self._i, -self.integrator_limit, self.integrator_limit))

        if self._prev_e is None:
            de = 0.0
        else:
            de = (e - self._prev_e) / max(self.dt, 1e-6)
        self._prev_e = e

        u = self.kp * e + self.ki * self._i + self.kd * de
        return float(u)


def _vec2(x, y):
    return np.array([float(x), float(y)], dtype=np.float32)


def _wrap_to_pi(a):
    a = float(a)
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return float(a)


# ============================================================
# Env
# ============================================================
class CarlaEnv:
    # Constants (from settings)
    show_preview = settings.SHOW_PREVIEW
    save_imgs = settings.SAVE_IMGS
    enable_spectator = settings.SPECTATOR
    MAX_STEER = settings.MAX_STEER
    MAX_THROTTLE_BRAKE = settings.MAX_THROTTLE_BRAKE
    THROTTLE_BRAKE_OFFSET = settings.THROTTLE_BRAKE_OFFSET

    def __init__(
        self,
        carla_town="Town04",
        max_npc_vehicles=10,
        desired_speed=65,  # km/h (kept for compatibility)
        max_stall_time=5,
        stall_speed=0.5,  # km/h
        seconds_per_episode=50,
        fps=10,
        server_port=4000,
        tm_port=8000,  # kept (but leader not TM-controlled here)
        verbose=False,
        pre_transform_image_height=128,
        pre_transform_image_width=256,
        fov=120,
        cam_x=1.3,
        cam_y=0.0,
        cam_z=1.75,
        cam_pitch=-15,
        lambda_r1=1.0,
        lambda_r2=0.3,
        lambda_r3=1.0,
        lambda_r4=0.005,
        lambda_r5=1.0,
        weather_presets=None,
        # ----------------------
        # Reward params (TTC gate + THW in-range + dv tracking + close shaping)
        # ----------------------
        ttc_safe=4.0,
        ttc_danger=2.0,
        alpha_ttc=1.25,
        ttc_beta=2.0,
        thw_center=2.0,
        thw_sigma=0.45,     # kept (not used by default; kept for compatibility)
        thw_floor=1.2,      # below this, headway is too small (extra penalty)
        thw_penalty_low=1.0,
        dv_scale=5.0,       # kept (not used by default; kept for compatibility)
        w_thw=1.2,          # weight for THW-in-range objective
        w_dv=0.6,           # weight for dv tracking (directional)
        w_ttc=2.0,          # weight for TTC risk penalty
        # NEW knobs (safe-following focus)
        w_close=0.45,       # weight for "close/open gap in the correct direction"
        w_act=0.15,         # weight for action shaping (brake rewarded in risky buckets)
        step_time_penalty=-0.01,   # small time penalty after warmup to avoid "do nothing"
        no_leader_penalty=-0.02,   # discourage losing leader/invalid CF
        offroute_penalty=-0.02,    # discourage leaving route corridor
        # dv desired tracking (key fix: do NOT punish dv>0 when THW is too large)
        dv_k=2.0,            # m/s per second headway error
        dv_des_max=6.0,      # cap desired dv magnitude
        dv_track_scale=3.0,  # how strict dv tracking is
        # THW target band (hard objective band)
        thw_good_lo=1.5,
        thw_good_hi=2.5,
        # THW out-of-band penalty scale (seconds to reach full -1)
        thw_out_scale=1.5,
    ):
        # ----------------------
        # Basic params
        # ----------------------
        self.curl_driving = False
        self._is_eval = False

        self.carla_town = carla_town
        self.max_npc_vehicles = max_npc_vehicles
        self.desired_speed = desired_speed
        self.max_stall_time = max_stall_time
        self.stall_speed = stall_speed
        self.seconds_per_episode = seconds_per_episode
        self.fps = fps
        self.server_port = server_port
        self.tm_port = tm_port
        self.dt = 1.0 / fps
        self.verbose = verbose

        self.im_height = pre_transform_image_height
        self.im_width = pre_transform_image_width
        self.fov = fov
        self.cam_x = cam_x
        self.cam_y = cam_y
        self.cam_z = cam_z
        self.cam_pitch = cam_pitch

        # Reward lambdas (kept for compatibility)
        self.lambda_r1 = lambda_r1
        self.lambda_r2 = lambda_r2
        self.lambda_r3 = lambda_r3
        self.lambda_r4 = lambda_r4  # collision intensity penalty scale
        self.lambda_r5 = lambda_r5

        self.weather_presets = (
            weather_presets if weather_presets is not None else settings.WEATHER_PRESETS
        )

        # ----------------------
        # Pure car-following mode
        # ----------------------
        self.pure_cf_mode = True

        # warm-up: fixed 2.5s, step() forces ego action during warmup
        self.start_acc_time = 2.5
        # start with neutral warmup action; initial ego velocity will be set at spawn
        self.warmup_action = np.array([0.0, 0.0], dtype=np.float32)
        self.warmup_steps = int(self.start_acc_time * self.fps)

        # ----------------------
        # Hard safety thresh
        # ----------------------
        self.g_min = 4.0  # m
        self.too_close_penalty = -60.0  # terminal penalty (FiT=4)  (stronger => safety)

        # ----------------------
        # CF metrics params
        # ----------------------
        self.gap_max = 120.0
        self.thw_max = 100.0
        # IMPORTANT: do NOT gate THW by speed (prevents "low-speed escape").
        self.thw_eps_v = 0.25  # m/s eps for THW denominator

        # TTC thresholds
        self.ttc_safe = float(ttc_safe)
        self.ttc_danger = float(ttc_danger)
        self.alpha_ttc = float(alpha_ttc)
        self.ttc_beta = float(ttc_beta)

        # THW objective band (the core goal)
        self.thw_center = float(thw_center)
        self.thw_good_lo = float(thw_good_lo)
        self.thw_good_hi = float(thw_good_hi)
        self.thw_out_scale = float(max(1e-6, thw_out_scale))
        self.thw_sigma = float(thw_sigma)  # kept for compatibility
        self.thw_floor = float(thw_floor)
        self.thw_penalty_low = float(thw_penalty_low)

        # dv (directional tracking)
        self.dv_k = float(dv_k)
        self.dv_des_max = float(dv_des_max)
        self.dv_track_scale = float(max(1e-6, dv_track_scale))
        self.dv_scale = float(dv_scale)  # kept for compatibility
        self.ttc_eps = 0.05  # kept for compatibility

        # close-gap memory
        self.close_clip_m = 1.5
        self.close_min_gap = 6.0
        self._prev_gap_for_close = float("nan")

        # weights
        self.w_thw = float(w_thw)
        self.w_dv = float(w_dv)
        self.w_ttc = float(w_ttc)
        self.w_close = float(w_close)
        self.w_act = float(w_act)

        self.step_time_penalty = float(step_time_penalty)
        self.no_leader_penalty = float(no_leader_penalty)
        self.offroute_penalty = float(offroute_penalty)

        # legacy placeholders kept for compatibility/debug
        self.enable_far_chase_boost = True
        self.far_chase_thw_start = 2.5
        self.far_chase_thw_full = 12.0
        self.far_chase_speed_target_ms = 15.0
        self.far_chase_throttle_bonus = 0.25
        self.far_chase_brake_penalty = 1.00
        self.far_chase_speed_weight = 1.00
        self.far_chase_clip = 1.0

        # ----------------------
        # Route & controllers
        # ----------------------
        self.route_step = 2.0  # meters per waypoint
        self.route_horizon_m = 1400.0
        self.route = []  # list[carla.Waypoint]

        # lateral control (Stanley-like)
        self.stanley_k = 1.2
        self.steer_lookahead_min = 6.0
        self.steer_lookahead_time = 0.6

        # leader longitudinal PID
        self.leader_speed_pid = PID(kp=0.45, ki=0.06, kd=0.02, dt=self.dt, integrator_limit=2.0)

        # optional: small throttle kick if leader stuck
        self.leader_kick_speed_eps = 0.25  # m/s
        self.leader_kick_throttle = 0.35
        self.leader_kick_max_steps = self.warmup_steps + max(1, int(0.5 * self.fps))
        self._leader_kick_count = 0

        # route indices
        self._ego_route_idx = 0
        self._leader_route_idx = 0

        # ----------------------
        # Actors & sensors states
        # ----------------------
        self.obs = None
        self.rgb_image = np.zeros((RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.uint8)

        self.FiT = 0
        self.episode_step = 0
        self.total_step = 0
        self.reset_counter = 0

        self.actor_list = []
        self.npc_vehicles_list = []
        self.collision_history = []

        # Leader handle
        self.leader_vehicle = None
        self._current_frame = None

        # per-episode metrics (warmup excluded)
        self._ttc_list = []
        self.dv_abs_list = []
        self.thw_list = []
        self.valid_cf_steps = 0
        self.thw_valid_steps = 0

        # NEW: "good following" counters (warmup excluded)
        self.thw_in_range_steps = 0
        self.ttc_safe_steps = 0
        self.safe_follow_steps = 0

        # robust validity diagnostics (warmup excluded)
        self.same_lane_steps = 0          # now means "same planned route corridor"
        self.offlane_steps = 0            # ego deviated from route corridor
        self.leader_missing_steps = 0     # leader not alive/usable
        self.junction_uncertain_steps = 0

        self._offlane_streak = 0
        self.offlane_streak_max = 0
        self._leader_missing_streak = 0
        self.leader_missing_streak_max = 0

        # Control display
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.abs_kmh = 0.0
        self.info = {}

        # ----------------------
        # Print environment info
        # ----------------------
        print(f"CARLA server port: {self.server_port}")
        print(f"CARLA traffic manager port: {self.tm_port}")
        print(f"CARLA_VERSION: {CARLA_VERSION_TUPLE}")

        # ----------------------
        # Server & client
        # ----------------------
        if os.name == "nt":
            self.server = CarlaServer(port=self.server_port, offscreen=False, sound=False)
        else:
            self.server = CarlaServer(port=self.server_port, offscreen=True, sound=True)
        self.server.launch(delay=20.0, retries=3)

        self.client = carla.Client("localhost", self.server_port)
        self.client.set_timeout(TIMEOUT)

        # World
        self.world = self.client.load_world(self.carla_town)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        # Synchronous world settings
        self.world_settings = self.world.get_settings()
        self.world_settings.synchronous_mode = True
        self.world_settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.world_settings)

        # Blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        # Route config from settings
        map_config = settings.map_config
        spawn_point_info = map_config[self.carla_town]
        ego_config = spawn_point_info["ego_config"]
        npc_config = spawn_point_info["npc_config"]  # kept for compatibility printing/assert

        # Ego spawn points
        self.ego_vehicle_possible_transforms = []
        for lane_id in ego_config["lanes"]:
            tf = self.map.get_waypoint_xodr(
                road_id=ego_config["road_id"], lane_id=lane_id, s=ego_config["start_s"]
            ).transform
            tf.location.z += SPAWN_HEIGHT
            self.ego_vehicle_possible_transforms.append(tf)

        # NPC spawn points (not used in pure_cf_mode, kept for old asserts/prints)
        self.npc_vehicle_possible_transforms = []
        for idx in range(len(npc_config["road_id"])):
            road_id = npc_config["road_id"][idx]
            start_lanes = npc_config["lanes"][idx]
            start_s = npc_config["start_s"][idx]
            npc_spawn_horizon = npc_config["max_s"][idx]
            npc_spawn_spacing = npc_config["spacing"][idx]

            distances = list(range(int(npc_spawn_horizon / npc_spawn_spacing + 1)))
            distances = [x * npc_spawn_spacing for x in distances]

            if road_id == ego_config["road_id"]:
                distances_to_remove = []
                for d in distances:
                    if (d < ego_config["start_s"] + npc_spawn_spacing) and (
                        d > ego_config["start_s"] - npc_spawn_spacing
                    ):
                        distances_to_remove.append(d)
                for d in distances_to_remove:
                    distances.remove(d)

            for npc_s in distances:
                for lane_id in start_lanes:
                    tf = self.map.get_waypoint_xodr(
                        road_id=road_id, lane_id=lane_id, s=npc_s
                    ).transform
                    tf.location.z += SPAWN_HEIGHT
                    self.npc_vehicle_possible_transforms.append(tf)

        print(
            f"[carla_env.py] Found {len(self.npc_vehicle_possible_transforms)} possible NPC vehicle spawn points for given configuration."
        )
        assert (
            len(self.npc_vehicle_possible_transforms) > self.max_npc_vehicles
        ), "Not enough NPC vehicle spawn points"

        # Ego vehicle bp
        self.ego_vehicle_bp = self.blueprint_library.filter("vehicle.tesla.model3")[0]

        # Leader vehicle bp
        self.leader_vehicle_bp = self.blueprint_library.filter("vehicle.audi.a2")[0]

        # Camera sensor
        self.camera_sensor_bp = self.blueprint_library.find("sensor.camera.rgb")
        self.camera_sensor_bp.set_attribute("image_size_x", f"{RENDER_WIDTH}")
        self.camera_sensor_bp.set_attribute("image_size_y", f"{RENDER_HEIGHT}")
        self.camera_sensor_bp.set_attribute("fov", f"{self.fov}")
        self.camera_sensor_bp.set_attribute("sensor_tick", "0.0")
        self.camera_sensor_transform = carla.Transform(
            carla.Location(x=self.cam_x, y=self.cam_y, z=self.cam_z),
            carla.Rotation(pitch=self.cam_pitch),
        )

        # Collision sensor
        self.collision_sensor_bp = self.blueprint_library.find("sensor.other.collision")

        # Spectator
        self.spectator = self.world.get_spectator()

        # Max episode steps
        assert isinstance(self.seconds_per_episode, int)
        assert isinstance(self.fps, int)
        self._max_episode_steps = int(self.seconds_per_episode * self.fps)

        # Save images
        if self.save_imgs:
            if os.path.exists("_out"):
                shutil.rmtree("_out")
            os.mkdir("_out")

        # Synchronous mode
        self.set_synchronous_mode(True)

        # Sensors
        self.camera_sensor = None
        self.collision_sensor = None
        self.camera_sensor_queue = None

        # corridor thresholds (route-projection robustness)
        # NOTE: Town04 lane width ~ 3.5m; use ~3.2 as "on-route" corridor
        self.route_on_dist = 3.2
        self.route_on_dist_leader = 3.4  # leader can drift slightly more

    # =========================================================
    # NEW: biased forward exploration action sampler (for train.py init_steps)
    # =========================================================
    def sample_action_forward(self):
        """
        Return an action biased to "move forward":
        - throttle_brake in [0.15, 0.85] (mostly positive => throttle)
        - steer = 0.0 (env controls lateral)
        This DOES NOT affect env unless train.py chooses to call it.
        """
        lo = 0.15 * float(self.MAX_THROTTLE_BRAKE)
        hi = 0.85 * float(self.MAX_THROTTLE_BRAKE)
        a0 = float(np.random.uniform(lo, hi))
        a1 = 0.0
        return np.array([a0, a1], dtype=np.float32)

    # =========================================================
    # Tick / frame alignment helpers
    # =========================================================
    def _drain_queue(self, q, max_items=200):
        if q is None:
            return
        n = 0
        try:
            while n < max_items:
                q.get_nowait()
                n += 1
        except Exception:
            pass

    def _tick_world(self):
        frame_id = self.world.tick(TIMEOUT)
        self._current_frame = int(frame_id)
        return int(frame_id)

    def collect_sensor_data(self, expected_frame=None, timeout=8.0):
        deadline = time.time() + float(timeout)
        last = None

        while time.time() < deadline:
            try:
                data = self.camera_sensor_queue.get(timeout=0.2)
                last = data

                if expected_frame is None:
                    self.process_camera_data(data)
                    return int(data.frame)

                f = int(data.frame)
                ef = int(expected_frame)

                if f < ef:
                    continue

                self.process_camera_data(data)
                return f

            except queue.Empty:
                continue

        raise Exception(
            f"Timeout while waiting for camera sensor data. expected_frame={expected_frame}, "
            f"last_frame={getattr(last, 'frame', None)}"
        )

    # =========================================================
    # Traffic light helpers (force green)
    # =========================================================
    def _force_vehicle_light_green_if_needed(self, veh: carla.Vehicle):
        try:
            if veh is None or (not getattr(veh, "is_alive", False)):
                return
            if veh.is_at_traffic_light():
                tl = veh.get_traffic_light()
                if tl is not None:
                    tl.set_state(carla.TrafficLightState.Green)
                    tl.set_green_time(999.0)
        except Exception:
            pass

    # =========================================================
    # Route building (deterministic at junction)
    # =========================================================
    def _choose_next_wp_deterministic(self, cur_wp: carla.Waypoint, step_m: float):
        cands = cur_wp.next(step_m)
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]

        cur_yaw = float(cur_wp.transform.rotation.yaw)
        best = None
        best_d = 1e9
        for w in cands:
            yaw = float(w.transform.rotation.yaw)
            dyaw = abs(_wrap_to_pi(math.radians(yaw - cur_yaw)))
            if dyaw < best_d:
                best_d = dyaw
                best = w
        return best if best is not None else cands[0]

    def _build_route_from_ego(self):
        self.route = []
        try:
            wp0 = self.map.get_waypoint(
                self.ego_vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception:
            return False

        self.route.append(wp0)

        n = int(self.route_horizon_m / max(self.route_step, 0.5))
        cur = wp0
        for _ in range(n):
            nxt = self._choose_next_wp_deterministic(cur, self.route_step)
            if nxt is None:
                break
            self.route.append(nxt)
            cur = nxt

        self._ego_route_idx = 0
        self._leader_route_idx = 0
        return len(self.route) > 10

    def _nearest_route_index(self, loc: carla.Location, start_idx: int, window: int = 60):
        if not self.route:
            return 0
        s = int(max(0, start_idx))
        e = int(min(len(self.route) - 1, start_idx + max(10, window)))
        px = float(loc.x)
        py = float(loc.y)

        best_i = s
        best_d2 = 1e18
        for i in range(s, e + 1):
            w = self.route[i]
            lx = float(w.transform.location.x)
            ly = float(w.transform.location.y)
            d2 = (lx - px) ** 2 + (ly - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return int(best_i)

    def _project_vehicle_to_route(self, vehicle: carla.Vehicle, base_idx: int, window: int, max_dist: float):
        """
        Project vehicle to nearest route waypoint in a local window.
        Return:
          idx, dist2d, on_route(bool), wp(idx)
        """
        if (vehicle is None) or (not getattr(vehicle, "is_alive", False)) or (not self.route):
            return base_idx, float("inf"), False, None

        try:
            loc = vehicle.get_location()
        except Exception:
            return base_idx, float("inf"), False, None

        idx = self._nearest_route_index(loc, base_idx, window=window)
        wp = self.route[idx]
        try:
            wloc = wp.transform.location
            dist2d = float(math.sqrt((loc.x - wloc.x) ** 2 + (loc.y - wloc.y) ** 2))
        except Exception:
            dist2d = float("inf")
        on_route = bool(np.isfinite(dist2d) and (dist2d <= float(max_dist)))
        return idx, dist2d, on_route, wp

    def _target_wp_ahead(self, vehicle: carla.Vehicle, base_idx: int, lookahead_m: float):
        if not self.route:
            return None, base_idx

        try:
            loc = vehicle.get_location()
        except Exception:
            return self.route[min(base_idx, len(self.route) - 1)], base_idx

        base_idx = self._nearest_route_index(loc, base_idx, window=80)
        step = max(self.route_step, 0.5)
        ahead_n = int(max(1, lookahead_m / step))
        tgt_idx = int(min(len(self.route) - 1, base_idx + ahead_n))
        return self.route[tgt_idx], base_idx

    # =========================================================
    # Lateral control (Stanley-like)
    # =========================================================
    def _vehicle_speed_ms(self, vehicle: carla.Vehicle):
        try:
            v = vehicle.get_velocity()
            return float(math.sqrt(v.x * v.x + v.y * v.y))
        except Exception:
            return 0.0

    def _stanley_steer(self, vehicle: carla.Vehicle, base_idx: int):
        v = self._vehicle_speed_ms(vehicle)
        lookahead = max(self.steer_lookahead_min, v * self.steer_lookahead_time)

        tgt_wp, base_idx = self._target_wp_ahead(vehicle, base_idx, lookahead)
        if tgt_wp is None:
            return 0.0, base_idx

        try:
            veh_tf = vehicle.get_transform()
            veh_loc = veh_tf.location
            veh_yaw = math.radians(float(veh_tf.rotation.yaw))
            fwd = _vec2(math.cos(veh_yaw), math.sin(veh_yaw))
            pos = _vec2(veh_loc.x, veh_loc.y)

            tgt_loc = tgt_wp.transform.location
            tgt = _vec2(tgt_loc.x, tgt_loc.y)

            to_tgt = tgt - pos
            dist = float(np.linalg.norm(to_tgt) + 1e-6)
            to_tgt_u = to_tgt / dist

            cross = float(fwd[0] * to_tgt_u[1] - fwd[1] * to_tgt_u[0])
            dot = float(fwd[0] * to_tgt_u[0] + fwd[1] * to_tgt_u[1])
            heading_err = float(math.atan2(cross, dot))

            cte = cross * dist
            stanley = float(math.atan2(self.stanley_k * cte, max(v, 1e-3)))

            steer = heading_err + stanley
            steer = float(np.clip(steer, -1.0, 1.0))
            steer = float(np.clip(steer, -self.MAX_STEER, self.MAX_STEER))
            return steer, base_idx
        except Exception:
            return 0.0, base_idx

    # =========================================================
    # Leader speed profile helper
    # =========================================================
    def _leader_target_speed_kmh(self, t_sec: float):
        try:
            t = float(t_sec)
            v0 = 30.0  # 起始
            v1 = 65.0  # 目标 (修改为65)
            t_acc = 40.0  # 加速时间

            if t <= 0.0:
                v_target = float(v0)
            elif t < t_acc:
                # 线性插值
                frac = t / float(t_acc)
                v_target = float(v0 + (v1 - v0) * frac)
            else:
                # 巡航
                v_target = float(v1)

            # [删除] 移除了 v_noise 正弦波
            return float(v_target)
        except Exception:
            return float(self.desired_speed)

    def _leader_apply_control(self, t_sec: float):
        # --- DO NOT change leader control logic ---
        if self.leader_vehicle is None or (not getattr(self.leader_vehicle, "is_alive", False)):
            return

        steer, self._leader_route_idx = self._stanley_steer(self.leader_vehicle, self._leader_route_idx)

        v_ms = self._vehicle_speed_ms(self.leader_vehicle)
        v_kmh = v_ms * 3.6
        v_ref_kmh = self._leader_target_speed_kmh(t_sec)

        e = float(v_ref_kmh - v_kmh)
        u = self.leader_speed_pid.step(e)

        throttle = float(np.clip(u, 0.0, 1.0))
        brake = float(np.clip(-u, 0.0, 1.0))

        if int(self.episode_step) <= int(self.leader_kick_max_steps):
            if float(v_ms) < float(self.leader_kick_speed_eps) and float(v_ref_kmh) > 1.0:
                throttle = float(max(throttle, self.leader_kick_throttle))
                brake = 0.0
                self._leader_kick_count += 1

        try:
            self.leader_vehicle.apply_control(
                carla.VehicleControl(
                    throttle=throttle,
                    steer=float(np.clip(steer, -self.MAX_STEER, self.MAX_STEER)),
                    brake=brake,
                    hand_brake=False,
                    reverse=False,
                    manual_gear_shift=False,
                )
            )
        except Exception:
            pass

    # =========================================================
    # Spawn helpers
    # =========================================================
    def _destroy_leader_if_exists(self):
        if self.leader_vehicle is None:
            return
        try:
            if self.leader_vehicle.is_alive:
                self.leader_vehicle.destroy()
        except Exception:
            pass
        self.leader_vehicle = None

    def _spawn_leader_same_lane_as_ego(self, gap_m: float, hard_retry: int = 25):
        """
        Spawn leader ahead using ego's current road/lane s+gap.
        NOTE: This is only for reset. During step, validity is route-projection based.
        """
        for _ in range(int(max(1, hard_retry))):
            self._destroy_leader_if_exists()

            try:
                wp_ego = self.map.get_waypoint(
                    self.ego_vehicle.get_location(),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                road_id = int(wp_ego.road_id)
                lane_id = int(wp_ego.lane_id)
                s_ego = float(wp_ego.s)
            except Exception:
                continue

            g0 = float(gap_m)
            min_gap = 20.1
            gap_candidates = [
                max(min_gap, g0),
                max(min_gap, g0 - 4.0),
                max(min_gap, g0 - 8.0),
                max(min_gap, g0 - 12.0),
            ]
            seen = set()
            gap_candidates = [x for x in gap_candidates if not (x in seen or seen.add(x))]

            for gap in gap_candidates:
                try:
                    s_spawn = s_ego + float(gap)
                    wp_spawn = self.map.get_waypoint_xodr(road_id=road_id, lane_id=lane_id, s=s_spawn)
                    if wp_spawn is None:
                        continue
                    spawn_tf = wp_spawn.transform
                    spawn_tf.location.z += SPAWN_HEIGHT

                    leader = self.world.try_spawn_actor(self.leader_vehicle_bp, spawn_tf)
                    if leader is None:
                        continue

                    self.leader_vehicle = leader
                    self.actor_list.append(leader)
                    self.npc_vehicles_list.append(leader)

                    # tick once so transform settles
                    _ = self._tick_world()

                    print(f"[LEADER] SPAWNED id={leader.id} gap≈{gap:.1f}m")
                    return True
                except Exception:
                    continue

        print("[LEADER] FAILED: cannot spawn leader ahead of ego (reset stage)")
        self.leader_vehicle = None
        return False

    # =========================================================
    # Core API
    # =========================================================
    def reset(self):
        self.destroy_all_actors()

        self.actor_list = []
        self.npc_vehicles_list = []
        self.collision_history = []

        # metrics reset
        self._ttc_list = []
        self.dv_abs_list = []
        self.thw_list = []
        self.kmh_tracker = []
        self.valid_cf_steps = 0
        self.thw_valid_steps = 0

        self.thw_in_range_steps = 0
        self.ttc_safe_steps = 0
        self.safe_follow_steps = 0

        # robustness reset
        self.same_lane_steps = 0
        self.offlane_steps = 0
        self.leader_missing_steps = 0
        self.junction_uncertain_steps = 0
        self._offlane_streak = 0
        self.offlane_streak_max = 0
        self._leader_missing_streak = 0
        self.leader_missing_streak_max = 0

        self._current_frame = None
        self.leader_vehicle = None
        self.info = {}

        self._leader_kick_count = 0
        self.leader_speed_pid.reset()

        # reset close-gap memory
        self._prev_gap_for_close = float("nan")

        # Warmup steps
        self.warmup_steps = int(self.start_acc_time * self.fps)
        self.leader_kick_max_steps = self.warmup_steps + max(1, int(0.5 * self.fps))

        # Weather
        self.weather_preset_idx = self.reset_counter % len(self.weather_presets)
        weather_preset = self.weather_presets[self.weather_preset_idx]
        weather_preset.sun_azimuth_angle = np.random.randint(30, 330)
        self.world.set_weather(weather_preset)

        # Spawn ego
        start_time = time.time()
        while True:
            try:
                self.ego_vehicle_transform = random.choice(self.ego_vehicle_possible_transforms)
                self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_bp, self.ego_vehicle_transform)
                # give ego an initial forward velocity to avoid cold-start stall
                try:
                    # As Phase2 cold-start matching: set ego initial speed to 8.33 m/s (~30 km/h)
                    # Use explicit world-frame target velocity per request.
                    try:
                        self.ego_vehicle.set_target_velocity(carla.Vector3D(x=8.33, y=0, z=0))
                    except Exception:
                        # fallback: compute forward direction and apply gentle velocity or throttle
                        try:
                            tf = self.ego_vehicle.get_transform()
                            yaw = math.radians(float(tf.rotation.yaw))
                            fx = math.cos(yaw)
                            fy = math.sin(yaw)
                            target_speed_ms = 8.33
                            vel = carla.Vector3D(x=fx * target_speed_ms, y=fy * target_speed_ms, z=0.0)
                            try:
                                self.ego_vehicle.set_target_velocity(vel)
                            except Exception:
                                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.12, steer=0.0, brake=0.0))
                        except Exception:
                            try:
                                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.12, steer=0.0, brake=0.0))
                            except Exception:
                                pass
                except Exception:
                    try:
                        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.12, steer=0.0, brake=0.0))
                    except Exception:
                        pass
                break
            except Exception:
                time.sleep(0.05)
            if time.time() - start_time > TIMEOUT:
                raise Exception("Timeout while waiting for ego vehicle to spawn")

        self.actor_list.append(self.ego_vehicle)

        # Spectator
        if self.enable_spectator:
            yaw = self.ego_vehicle_transform.rotation.yaw * (math.pi / 180)
            dist = -7.5
            dx = dist * math.cos(yaw)
            dy = dist * math.sin(yaw)
            self.spectator.set_transform(
                carla.Transform(
                    self.ego_vehicle_transform.location + carla.Location(x=dx, y=dy, z=5),
                    carla.Rotation(yaw=self.ego_vehicle_transform.rotation.yaw, pitch=-25),
                )
            )

        # Let ego fall to ground
        delta_t = math.sqrt((2 * SPAWN_HEIGHT) / G) + 0.75
        nb_steps = math.ceil(delta_t / self.dt)
        for _ in range(nb_steps):
            self._tick_world()
            time.sleep(2 * self.dt)

        # Build route from ego
        ok_route = self._build_route_from_ego()
        if not ok_route:
            print("[ROUTE] WARNING: route build failed (will fallback to steer=0).")

        # Print ego lane
        try:
            wp_ego = self.map.get_waypoint(
                self.ego_vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            print(f"[EGO] road={wp_ego.road_id} lane={wp_ego.lane_id}")
        except Exception:
            pass

        # Spawn leader at reset
        gap0 = float(np.random.uniform(20.0, 30.0))
        self._spawn_leader_same_lane_as_ego(gap0, hard_retry=25)

        # Spawn camera
        self.camera_sensor = self.world.spawn_actor(
            self.camera_sensor_bp, self.camera_sensor_transform, attach_to=self.ego_vehicle
        )
        self.actor_list.append(self.camera_sensor)
        self.camera_sensor_queue = queue.Queue()
        self.camera_sensor.listen(self.camera_sensor_queue.put)

        # Spawn collision sensor
        self.collision_sensor = self.world.spawn_actor(
            self.collision_sensor_bp, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.process_collision_data(event))

        # Reset camera queue, tick once, align
        self._drain_queue(self.camera_sensor_queue, max_items=200)
        frame0 = self._tick_world()
        _ = self.collect_sensor_data(expected_frame=frame0, timeout=8.0)

        # Administration
        self.reset_counter += 1
        self.episode_step = 0
        self.total_step = self.total_step
        self.stall_counter = 0
        self.abs_kmh = 0.0
        self.FiT = 0

        # reset indices
        self._ego_route_idx = 0
        self._leader_route_idx = 0

        return self.obs

    def _process_action_longitudinal(self, action):
        """
        action: np.array([throttle_brake, steer]) but steer ignored for RL (env controls lateral)
        """
        action = np.array(action, dtype=np.float32).copy()
        action[0] = np.clip(action[0], -self.MAX_THROTTLE_BRAKE, self.MAX_THROTTLE_BRAKE)
        action[0] = np.clip(
            action[0] + self.THROTTLE_BRAKE_OFFSET,
            -self.MAX_THROTTLE_BRAKE,
            self.MAX_THROTTLE_BRAKE,
        )

        throttle = float(np.max([action[0], 0.0]))
        brake = float(-np.min([action[0] / (1 - self.THROTTLE_BRAKE_OFFSET), 0.0]))
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))
        return action, throttle, brake

    def step(self, action):
        # force green lights
        self._force_vehicle_light_green_if_needed(self.ego_vehicle)
        if self.leader_vehicle is not None:
            self._force_vehicle_light_green_if_needed(self.leader_vehicle)

        try:
            if int(self.episode_step) < 50:
                action = np.array([0.4, 0.0], dtype=np.float32)
        except Exception:
            action = np.array([0.4, 0.0], dtype=np.float32)

        # warm-up: force ego action
        in_warmup = False
        try:
            in_warmup = (int(self.episode_step) < int(self.warmup_steps))
        except Exception:
            in_warmup = False
        if in_warmup:
            action = self.warmup_action.copy()

        # leader control (route + speed profile)
        t_sec = float(self.episode_step * self.dt)
        self._leader_apply_control(t_sec)

        # ego lateral control (ignore action[1])
        ego_steer, self._ego_route_idx = self._stanley_steer(self.ego_vehicle, self._ego_route_idx)
        self.steer = float(ego_steer)

        # ego longitudinal from action[0]
        action, self.throttle, self.brake = self._process_action_longitudinal(action)

        # Apply ego control
        self.ego_vehicle.apply_control(
            carla.VehicleControl(
                throttle=float(self.throttle),
                steer=float(np.clip(self.steer, -self.MAX_STEER, self.MAX_STEER)),
                brake=float(self.brake),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
            )
        )

        # tick + frame-aligned sensor
        frame_id = self._tick_world()
        self.episode_step += 1
        self.total_step += 1
        _ = self.collect_sensor_data(expected_frame=frame_id, timeout=8.0)

        # -------------------------
        # compute per-step metrics (store into self.info)
        # -------------------------
        info_step = {}

        # speeds
        v_ego_ms = self._vehicle_speed_ms(self.ego_vehicle)
        self.abs_kmh = float(3.6 * v_ego_ms)

        leader_alive = (self.leader_vehicle is not None) and getattr(self.leader_vehicle, "is_alive", False)
        v_lead_ms = float("nan")
        if leader_alive:
            v_lead_ms = self._vehicle_speed_ms(self.leader_vehicle)

        # route projection (ROBUST)
        ego_idx, ego_d, ego_on, ego_wp = self._project_vehicle_to_route(
            self.ego_vehicle, self._ego_route_idx, window=90, max_dist=self.route_on_dist
        )
        self._ego_route_idx = int(ego_idx)

        lead_idx, lead_d, lead_on, lead_wp = self._project_vehicle_to_route(
            self.leader_vehicle, self._leader_route_idx, window=120, max_dist=self.route_on_dist_leader
        )
        self._leader_route_idx = int(lead_idx)

        # leader ahead on route?
        leader_ahead = bool(lead_on and ego_on and (int(lead_idx) > int(ego_idx)))

        # gap by route index diff (primary)
        gap_route = float("nan")
        if leader_ahead:
            gap_route = float((int(lead_idx) - int(ego_idx)) * float(self.route_step))
            gap_route = float(np.clip(gap_route, 0.0, float(self.gap_max)))

        # fallback: if not leader_ahead but leader alive, use Euclidean gap
        gap_dist2d = float("nan")
        if leader_alive:
            try:
                ego_loc = self.ego_vehicle.get_location()
                lead_loc = self.leader_vehicle.get_location()
                gap_dist2d = float(math.sqrt((lead_loc.x - ego_loc.x) ** 2 + (lead_loc.y - ego_loc.y) ** 2))
            except Exception:
                gap_dist2d = float("nan")

        # junction uncertain (route wp)
        junc_unc = False
        try:
            if ego_wp is not None and bool(getattr(ego_wp, "is_junction", False)):
                junc_unc = True
            if lead_wp is not None and bool(getattr(lead_wp, "is_junction", False)):
                junc_unc = True
        except Exception:
            junc_unc = False

        # define "same_lane" robustly as "both on planned route corridor"
        same_lane = bool(ego_on and lead_on)

        # define leader_exists/valid_cf robustly
        use_gap = float("nan")
        leader_exists = False
        if leader_alive and same_lane and leader_ahead and np.isfinite(gap_route) and gap_route > 0.0:
            use_gap = float(gap_route)
            leader_exists = True
        elif leader_alive and np.isfinite(gap_dist2d) and gap_dist2d > 0.0:
            if float(gap_dist2d) <= float(self.gap_max):
                use_gap = float(gap_dist2d)
                leader_exists = True

        # dv/ttc/thw
        dv = float("nan")
        ttc = float("nan")
        thw = float("nan")

        if leader_exists and np.isfinite(v_lead_ms) and np.isfinite(use_gap):
            dv = float(v_ego_ms - v_lead_ms)

            rel = float(v_ego_ms - v_lead_ms)
            if (not np.isfinite(rel)) or rel <= 0.0:
                ttc = float(MAX_TTC)
            else:
                ttc = float(use_gap / max(rel, 1e-6))
            ttc = float(min(max(0.0, ttc), MAX_TTC))

            denom_v = max(float(self.thw_eps_v), float(v_ego_ms))
            thw = float(use_gap / max(denom_v, 1e-6))
            thw = float(min(max(0.0, thw), float(self.thw_max)))

        # previous gap for close shaping (warmup excluded)
        in_warmup_now = (int(self.episode_step) <= int(self.warmup_steps))
        prev_gap = float(self._prev_gap_for_close) if np.isfinite(self._prev_gap_for_close) else float("nan")
        if (not in_warmup_now) and leader_exists and np.isfinite(use_gap):
            self._prev_gap_for_close = float(use_gap)

        # info step fields
        info_step.update({
            "step_leader_alive": bool(leader_alive),
            "step_same_lane": bool(same_lane),
            "step_leader_exists": bool(leader_exists),
            "step_junction_uncertain": bool(junc_unc),

            "step_gap": float(use_gap) if np.isfinite(use_gap) else float("nan"),
            "step_prev_gap": float(prev_gap) if np.isfinite(prev_gap) else float("nan"),
            "step_v_lead": float(v_lead_ms) if np.isfinite(v_lead_ms) else float("nan"),
            "step_v_ego_ms": float(v_ego_ms),

            "step_dv": float(dv) if np.isfinite(dv) else float("nan"),
            "step_ttc": float(ttc) if np.isfinite(ttc) else float("nan"),
            "step_thw": float(thw) if np.isfinite(thw) else float("nan"),

            # route robustness diagnostics
            "step_ego_on_route": bool(ego_on),
            "step_lead_on_route": bool(lead_on),
            "step_leader_ahead": bool(leader_ahead),
            "step_ego_route_dist": float(ego_d) if np.isfinite(ego_d) else float("nan"),
            "step_lead_route_dist": float(lead_d) if np.isfinite(lead_d) else float("nan"),

            "leader_kick_count": int(self._leader_kick_count),
        })

        # bookkeep metrics (warmup excluded)
        if not in_warmup_now:
            if leader_exists:
                self.valid_cf_steps += 1
            if same_lane:
                self.same_lane_steps += 1
            if junc_unc:
                self.junction_uncertain_steps += 1

            # offlane / missing streaks
            if not ego_on:
                self.offlane_steps += 1
                self._offlane_streak += 1
                self.offlane_streak_max = max(self.offlane_streak_max, self._offlane_streak)
            else:
                self._offlane_streak = 0

            if not leader_alive:
                self.leader_missing_steps += 1
                self._leader_missing_streak += 1
                self.leader_missing_streak_max = max(self.leader_missing_streak_max, self._leader_missing_streak)
            else:
                self._leader_missing_streak = 0

            # CF value lists only when leader exists
            if leader_exists:
                if np.isfinite(dv):
                    self.dv_abs_list.append(abs(float(dv)))
                if np.isfinite(thw):
                    self.thw_list.append(float(thw))
                    self.thw_valid_steps += 1
                if np.isfinite(ttc):
                    self._ttc_list.append(float(ttc))

                # NEW: good-following counters
                if np.isfinite(ttc) and float(ttc) >= float(self.ttc_safe):
                    self.ttc_safe_steps += 1
                if np.isfinite(thw) and (float(self.thw_good_lo) <= float(thw) <= float(self.thw_good_hi)):
                    self.thw_in_range_steps += 1
                if (np.isfinite(ttc) and float(ttc) >= float(self.ttc_safe)) and (
                    np.isfinite(thw) and (float(self.thw_good_lo) <= float(thw) <= float(self.thw_good_hi))
                ):
                    self.safe_follow_steps += 1

        # update self.info base
        if self.info is None or not isinstance(self.info, dict):
            self.info = {}
        self.info.update(info_step)

        # -------------------------
        # reward / done / info
        # -------------------------
        reward, done, info = self.reward_function(action)

        # time limit (FiT=2)
        if (not done) and (self.episode_step * self.dt >= self.seconds_per_episode):
            done = True
            self.FiT = 2
            info["FiT"] = self.FiT

        # -------------------------
        # Aggregate metrics for logging
        # -------------------------
        if len(self._ttc_list) == 0:
            min_TTC = float("nan")
            TTC5 = float("nan")
        else:
            try:
                min_TTC = float(np.nanmin(self._ttc_list))
                TTC5 = float(np.nanpercentile(self._ttc_list, 5))
            except Exception:
                min_TTC = float("nan")
                TTC5 = float("nan")

        info["min_TTC"] = min_TTC
        info["TTC5"] = TTC5

        try:
            info["MAE_dv"] = float(np.mean(self.dv_abs_list)) if len(self.dv_abs_list) > 0 else float("nan")
        except Exception:
            info["MAE_dv"] = float("nan")

        if len(self.thw_list) > 0:
            try:
                info["THW_mean"] = float(np.mean(self.thw_list))
            except Exception:
                info["THW_mean"] = float("nan")
            try:
                info["THW_p50"] = float(np.percentile(self.thw_list, 50))
            except Exception:
                info["THW_p50"] = float("nan")
            try:
                info["THW_p95"] = float(np.percentile(self.thw_list, 95))
            except Exception:
                info["THW_p95"] = float("nan")
        else:
            info["THW_mean"] = float("nan")
            info["THW_p50"] = float("nan")
            info["THW_p95"] = float("nan")

        denom = max(1, int(self.episode_step - self.warmup_steps))
        info["valid_cf_steps"] = int(self.valid_cf_steps)
        info["valid_cf_ratio"] = float(self.valid_cf_steps / denom) if denom > 0 else float("nan")

        info["THW_valid_steps"] = int(self.thw_valid_steps)
        info["THW_valid_ratio"] = float(self.thw_valid_steps / denom) if denom > 0 else float("nan")

        # NEW: key KPI ratios
        info["THW_in_range_steps"] = int(self.thw_in_range_steps)
        info["THW_in_range_ratio"] = float(self.thw_in_range_steps / denom) if denom > 0 else float("nan")
        info["TTC_safe_steps"] = int(self.ttc_safe_steps)
        info["TTC_safe_ratio"] = float(self.ttc_safe_steps / denom) if denom > 0 else float("nan")
        info["safe_follow_steps"] = int(self.safe_follow_steps)
        info["safe_follow_ratio"] = float(self.safe_follow_steps / denom) if denom > 0 else float("nan")

        # robust diagnostics
        info["same_lane_steps"] = int(self.same_lane_steps)
        info["offlane_steps"] = int(self.offlane_steps)
        info["leader_missing_steps"] = int(self.leader_missing_steps)
        info["junction_uncertain_steps"] = int(self.junction_uncertain_steps)

        info["offlane_streak_max"] = int(self.offlane_streak_max)
        info["leader_missing_streak_max"] = int(self.leader_missing_streak_max)

        info["same_lane_ratio"] = float(self.same_lane_steps / denom) if denom > 0 else float("nan")
        info["junction_uncertain_ratio"] = float(self.junction_uncertain_steps / denom) if denom > 0 else float("nan")

        info["leader_kick_count"] = int(self._leader_kick_count)

        # ensure step fields included
        for k, v in info_step.items():
            info[k] = v

        self.info = info
        return self.obs, float(reward), bool(done), info

    # =========================================================
    # Reward function (ONLY read from self.info / self.throttle/brake/abs_kmh)
    # =========================================================
    def reward_function(self, action):
        done = False
        reward = 0.0
        info_src = self.info if isinstance(self.info, dict) else {}

        # read state with safe conversions
        leader_exists = bool(info_src.get("step_leader_exists", False))
        gap = float(info_src.get("step_gap", float("nan")))
        v_ego_ms = float(info_src.get("step_v_ego_ms", 0.0))
        v_lead_ms = float(info_src.get("step_v_lead", float("nan")))
        v_ego_kmh = float(v_ego_ms) * 3.6

        # Collision handling (highest priority)
        if len(self.collision_history) > 0:
            try:
                ev = self.collision_history[0]
                impulse = getattr(ev, "normal_impulse", None)
                if impulse is not None:
                    jx = float(getattr(impulse, "x", 0.0))
                    jy = float(getattr(impulse, "y", 0.0))
                    jz = float(getattr(impulse, "z", 0.0))
                    j_mag = float(math.sqrt(jx * jx + jy * jy + jz * jz))
                else:
                    j_mag = float("inf")
            except Exception:
                j_mag = float("inf")

            # filter out tiny bumps/noise
            if np.isfinite(j_mag) and (j_mag > 100.0):
                pen = -np.clip(max(70.0, j_mag * 0.005), 70.0, 200.0)
                self.FiT = 1
                done = True
                out_info = info_src
                out_info.update({"collision_impulse_mag": j_mag, "step_r_collision": float(pen), "FiT": int(self.FiT)})
                self.info = out_info
                return float(pen), bool(done), out_info

        # initialize terms
        R_speed_base = 0.0
        R_thw = 0.0
        R_move = 0.0
        R_smooth = 0.0

        if leader_exists and np.isfinite(gap) and np.isfinite(v_ego_ms) and np.isfinite(v_lead_ms):
            # base speed alignment
            v_lead_kmh = float(v_lead_ms) * 3.6
            diff = abs(v_ego_kmh - v_lead_kmh)
            R_speed_base = 1.0 - (diff / 40.0)

            # smooth turtle linear penalty (v_min=20km/h)
            v_min = 20.0
            if v_ego_kmh < v_min:
                R_speed_base -= 0.2 * (v_min - v_ego_kmh)
            R_speed_base = float(np.clip(R_speed_base, -5.0, 5.0))

            # compute current_thw safely
            try:
                denom_v = max(float(self.thw_eps_v), float(v_ego_ms))
                current_thw = float(gap) / max(denom_v, 1e-6)
            except Exception:
                current_thw = float("nan")

            # --- Refined THW logic: continuous across (0, inf) ---
            if np.isfinite(current_thw):
                if current_thw < 1.0:
                    # continuous linear penalty from 1.0->0.0 mapped to 0 -> -2.5
                    R_thw = -2.5 * (1.0 - current_thw)
                elif 1.0 <= current_thw <= 4.0:
                    safety_base = 0.5
                    precision = 1.5 * math.exp(-0.5 * ((current_thw - 2.0) / 0.2) ** 2)
                    R_thw = safety_base + precision
                else:
                    R_thw = 0.0

            # R_move: only when gap large and ego is faster
            try:
                target_gap = max(2.0 * float(v_ego_ms), 15.0)
                if float(gap) > float(target_gap):
                    dv = float(v_ego_ms - v_lead_ms)
                    if dv > 0.0:
                        dv_cap = 6.0
                        R_move = float(min(dv, dv_cap) / dv_cap)
                    else:
                        R_move = 0.0
                else:
                    R_move = 0.0
            except Exception:
                R_move = 0.0

            # action smoothness
            try:
                a0 = float(action[0])
            except Exception:
                a0 = 0.0
            R_smooth = -0.1 * (a0 ** 2)

            # extra near-gap warning (non-terminal)
            if float(gap) < 5.0:
                # deduct 0.5 from total THW-related score to emphasize safety
                R_thw -= 0.5

        else:
            R_speed_base = float(self.no_leader_penalty)

        # final composition per spec
        reward = 1.0 * float(R_speed_base) + 2.5 * float(R_thw) + float(R_move) + float(R_smooth)

        # stall termination preserved
        try:
            if (self.episode_step >= 50) and (float(self.abs_kmh) < float(self.stall_speed)):
                self.stall_counter += 1
            else:
                self.stall_counter = 0
        except Exception:
            self.stall_counter = 0

        if (not done) and (self.stall_counter * self.dt >= float(self.max_stall_time)):
            self.FiT = 3
            done = True
            reward += -20.0

        # logging
        try:
            self.kmh_tracker.append(float(self.abs_kmh))
        except Exception:
            try:
                self.kmh_tracker = [float(self.abs_kmh)]
            except Exception:
                pass

        out_info = info_src
        try:
            mean_kmh = float(np.mean(self.kmh_tracker)) if len(self.kmh_tracker) > 0 else float("nan")
        except Exception:
            mean_kmh = float("nan")
        try:
            max_kmh = float(np.max(self.kmh_tracker)) if len(self.kmh_tracker) > 0 else float("nan")
        except Exception:
            max_kmh = float("nan")

        out_info.update({
            "step_r_speed": float(R_speed_base),
            "step_r_thw": float(R_thw),
            "step_r_move": float(R_move),
            "step_r_act": float(R_smooth),
            "FiT": int(self.FiT),
            "mean_kmh": mean_kmh,
            "max_kmh": max_kmh,
            "brake_sum": float(getattr(self, "brake_sum", 0.0)),
        })
        self.info = out_info

        return float(reward * 0.1), bool(done), self.info

    # =========================================================
    # Spaces
    # =========================================================
    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=0.0, high=255.0, shape=(3, self.im_height, self.im_width), dtype=np.uint8
        )

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=np.array([-self.MAX_THROTTLE_BRAKE, -self.MAX_STEER], dtype=np.float32),
            high=np.array([self.MAX_THROTTLE_BRAKE, self.MAX_STEER], dtype=np.float32),
            dtype=np.float32,
        )

    # =========================================================
    # Sync mode
    # =========================================================
    def set_synchronous_mode(self, synchronous):
        self.world_settings.synchronous_mode = synchronous
        if synchronous:
            self.world_settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.world_settings)

    # =========================================================
    # Sensor callbacks
    # =========================================================
    def process_camera_data(self, carla_im_data):
        raw_image = np.array(carla_im_data.raw_data)
        bgra = raw_image.reshape((RENDER_HEIGHT, RENDER_WIDTH, -1))
        bgr = bgra[:, :, :3]
        self.rgb_image = bgr[:, :, ::-1]
        obs = cv2.resize(self.rgb_image, (self.im_width, self.im_height), interpolation=cv2.INTER_AREA)

        if self.show_preview:
            cv2.imshow("", obs[:, :, ::-1])
            cv2.waitKey(1)

        if self.save_imgs:
            cv2.imwrite(
                os.path.join("_out", f"im_{self.reset_counter}_{self.episode_step}.png"),
                obs[:, :, ::-1],
            )

        obs = np.transpose(obs, (2, 0, 1))
        if self.save_imgs:
            np.save(os.path.join("_out", f"im_{self.reset_counter}_{self.episode_step}.npy"), obs)

        self.obs = obs

    def process_collision_data(self, event):
        self.collision_history.append(event)

    # =========================================================
    # Misc
    # =========================================================
    def seed(self, seed):
        random.seed(seed)
        self._np_random = np.random.RandomState(seed)

    def destroy_all_actors(self):
        if self.verbose:
            print("destroying actors")

        try:
            if self.camera_sensor is not None:
                self.camera_sensor.stop()
                self.camera_sensor.destroy()
            if self.collision_sensor is not None:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
        except Exception:
            print("[carla_env.py] No sensors to destroy or error destroying sensors.")

        if len(self.actor_list) != 0:
            try:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            except Exception:
                for a in self.actor_list:
                    try:
                        a.destroy()
                    except Exception:
                        pass

        self.actor_list = []
        self.npc_vehicles_list = []
        self.leader_vehicle = None

        self.camera_sensor = None
        self.collision_sensor = None
        self.camera_sensor_queue = None

        if self.verbose:
            print("done.\n\n")

    def deactivate(self):
        self.set_synchronous_mode(False)
        self.destroy_all_actors()
        self.server.kill()

    def render(self):
        frame = self.rgb_image.copy()

        bar_width = 200
        bar_height = 20
        bar_x = 10
        throttle_y = 30
        brake_y = 60
        steering_y = 90
        bar_color = (49, 61, 92)
        text_settings = (cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        throttle_width = int(bar_width * float(self.throttle))
        brake_width = int(bar_width * float(self.brake))
        steering_width = int(bar_width * (float(self.steer) / max(self.MAX_STEER, 1e-6)) / 2)

        cv2.rectangle(frame, (bar_x, throttle_y), (bar_x + throttle_width, throttle_y + bar_height), bar_color, -1)
        cv2.rectangle(frame, (bar_x, throttle_y), (bar_x + bar_width, throttle_y + bar_height), bar_color, 2)

        cv2.rectangle(frame, (bar_x, brake_y), (bar_x + brake_width, brake_y + bar_height), bar_color, -1)
        cv2.rectangle(frame, (bar_x, brake_y), (bar_x + bar_width, brake_y + bar_height), bar_color, 2)

        if self.steer > 0:
            cv2.rectangle(
                frame,
                (bar_x + int(bar_width / 2), steering_y),
                (bar_x + int(bar_width / 2) + steering_width, steering_y + bar_height),
                bar_color,
                -1,
            )
        else:
            cv2.rectangle(
                frame,
                (bar_x + int(bar_width / 2) + steering_width, steering_y),
                (bar_x + int(bar_width / 2), steering_y + bar_height),
                bar_color,
                -1,
            )
        cv2.rectangle(frame, (bar_x, steering_y), (bar_x + bar_width, steering_y + bar_height), bar_color, 2)
        cv2.rectangle(
            frame,
            (bar_x + int(bar_width / 2) - 1, steering_y - 1),
            (bar_x + int(bar_width / 2) + 1, steering_y + bar_height + 1),
            (255, 255, 255),
            -1,
        )

        cv2.putText(frame, "Throttle", (bar_x + bar_width + 10, throttle_y + bar_height - 3), *text_settings)
        cv2.putText(frame, "Brake", (bar_x + bar_width + 10, brake_y + bar_height - 3), *text_settings)
        cv2.putText(frame, "Steering(LatCtrl)", (bar_x + bar_width + 10, steering_y + bar_height - 3), *text_settings)

        x = frame.shape[1] - 230
        mode_settings = (cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "Mode: Pure Longitudinal CF", (x, 30), *mode_settings)

        if self.info is not None and isinstance(self.info, dict) and all(
            k in self.info for k in ["r1", "r2", "r3", "r4", "r5", "mean_kmh", "max_kmh"]
        ):
            cv2.putText(frame, "Cumulative reward", (x, 60), *text_settings)
            cv2.putText(frame, f"r1: {self.info['r1']:.4f}", (x, 90), *text_settings)
            cv2.putText(frame, f"r2: {self.info['r2']:.4f}", (x, 120), *text_settings)
            cv2.putText(frame, f"r3: {self.info['r3']:.4f}", (x, 150), *text_settings)
            cv2.putText(frame, f"r4: {self.info['r4']:.4f}", (x, 180), *text_settings)
            cv2.putText(frame, f"r5: {self.info['r5']:.4f}", (x, 210), *text_settings)

            r = self.info["r1"] + self.info["r2"] + self.info["r3"] + self.info["r4"] + self.info["r5"]
            if "r_lane" in self.info:
                r += self.info["r_lane"]
            if "r_center" in self.info:
                r += self.info["r_center"]

            cv2.putText(frame, f"Total: {r:.2f}", (x, 240), *text_settings)
            cv2.putText(frame, "-------------", (x, 270), *text_settings)
            cv2.putText(frame, f"Mean km/h: {self.info['mean_kmh']:.1f}", (x, 300), *text_settings)
            cv2.putText(frame, f"Max  km/h: {self.info['max_kmh']:.1f}", (x, 330), *text_settings)
            cv2.putText(frame, f"Cur  km/h: {self.abs_kmh:.1f}", (x, 360), *text_settings)

            try:
                cv2.putText(frame, f"FiT: {int(self.info.get('FiT', 0))}", (x, 390), *text_settings)
                cv2.putText(frame, f"Gap: {float(self.info.get('step_gap', float('nan'))):.2f} m", (x, 420), *text_settings)
                cv2.putText(frame, f"THW: {float(self.info.get('step_thw', float('nan'))):.2f} s", (x, 450), *text_settings)
                cv2.putText(frame, f"TTC: {float(self.info.get('step_ttc', float('nan'))):.2f} s", (x, 480), *text_settings)
                cv2.putText(frame, f"KickCnt: {int(self.info.get('leader_kick_count', 0))}", (x, 510), *text_settings)
                if "R_close" in self.info:
                    cv2.putText(frame, f"R_close: {float(self.info.get('R_close', 0.0)):.2f}", (x, 540), *text_settings)
                if "ttc_bucket" in self.info:
                    cv2.putText(frame, f"TTC_bkt: {str(self.info.get('ttc_bucket', 'na'))}", (x, 570), *text_settings)
            except Exception:
                pass

        return frame
