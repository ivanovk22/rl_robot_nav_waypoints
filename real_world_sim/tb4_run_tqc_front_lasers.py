#!/usr/bin/env python3
import os
import math
import argparse
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from irobot_create_msgs.srv import ResetPose



def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def yaw_from_quat(x, y, z, w, base_yaw):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(math.atan2(siny_cosp, cosy_cosp))
    new_yaw = wrap_to_pi(yaw + base_yaw)
    return new_yaw

def translate_position(x_local, y_local, global_x, global_y, base_yaw):
    x_global = global_x + math.cos(base_yaw) * x_local - math.sin(base_yaw) * y_local
    y_global = global_y + math.sin(base_yaw) * x_local + math.cos(base_yaw) * y_local
    return x_global, y_global




# Stable-Baselines3 (+ contrib)
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm  # type: ignore
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
from policies.custompolicy import CombinedFeatureExtractor


try:
    from sb3_contrib import TQC  # noqa: F401
    _HAS_TQC = True
except Exception:
    TQC = None  # type: ignore
    _HAS_TQC = False

try:
    import gymnasium as gym
except Exception:
    import gymnasium as gym  # require gymnasium in your venv

# Optional plotting (import lazily when enabled)
try:
    import matplotlib
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None


print_one = 0

@dataclass
class ControlLimits:
    max_lin: float = 0.4
    max_ang: float = 1.0


@dataclass
class LidarParams:
    """
    Here I should change the Lidar Parameters
    """


    num_rays: int = 180
    lidar_max: float = 30.0
    inv_alpha: float = 0.6
    dist_floor: float = 0.05
    k_bins: int = 24
    short_r: float = 1.0


@dataclass
class GoalParams:
    x: float = 5.0
    y: float = 0.0
    success_radius: float = 0.5


class DummyObsEnv(gym.Env):
    metadata = {"render.modes": []}
    def __init__(self, obs_dim: int = 146):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
    def reset(self, *args, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
    def step(self, action):
        return self.reset()[0], 0.0, True, False, {}


# ───────r────────────────── SB3 model auto-loader ─────────────────────────

ALGO_MAP = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C, "DQN": DQN}
if _HAS_TQC:
    ALGO_MAP["TQC"] = TQC  # type: ignore

def _load_sb3_model(model_path: str, trainer_hint: Optional[str], device: str = "cpu") -> BaseAlgorithm:
    custom_objects = {
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0
    }
    # 1) If user hints the algo, trust it
    if trainer_hint:
        algo_key = trainer_hint.strip().upper()
        if algo_key in ALGO_MAP and ALGO_MAP[algo_key] is not None:
            return ALGO_MAP[algo_key].load(model_path, device=device, custom_objects=custom_objects, print_system_info=False)  # type: ignore
        raise RuntimeError(f"Unknown --trainer '{trainer_hint}'. Available: {sorted(ALGO_MAP.keys())}")
    # 2) Try to read algo from saved JSON
    try:
        data, params, pytvars = load_from_zip_file(model_path, device=device, custom_objects=custom_objects, print_system_info=False)
        algo_key = (data.get("algo") or data.get("algorithm") or "").upper()
        if algo_key in ALGO_MAP and ALGO_MAP[algo_key] is not None:
            return ALGO_MAP[algo_key].load(model_path, device=device, custom_objects=custom_objects, print_system_info=False)  # type: ignore
    except Exception:
        pass
    # 3) Last resort: try TQC if available
    if _HAS_TQC:
        return TQC.load(model_path, device=device, custom_objects=custom_objects, print_system_info=False)  # type: ignore
    raise RuntimeError("Cannot determine algorithm from model zip and no --trainer provided (and TQC not available).")


# ───────────────────────── LiDAR helpers (training-aligned) ─────────────────────────

def _sanitize_ranges(scan: LaserScan, cap: float) -> np.ndarray:
    """
        Here the logic for the beans has to be changed with the one that is used in the trained model:
        Test_Linux_noise5M for now. If the logic(about how you process the lidar scans) in the training part
        is changed, you then have to change it here as well. THIS COMMENT MAYBE IS NOT CORRECT
    """

    r = np.asarray(scan.ranges, dtype=np.float32)


    L = len(r)
    
    K = L // 2 # Integer division for 1/4th, which is 270
    # 3. Calculate the pivot index where the cut is made
    # pivot_index = L - K # 1080 - 270 = 810
    # 4. Perform the rotation using slicing
    #    - original_array[pivot_index:]  -> The last K elements (the new front)
    #    - original_array[:pivot_index]  -> The first L-K elements (the new back)
    new_front = r[:K]  
    

    r = new_front.copy()

    # pool_size = 6

    # # 1. Ensure the array is compatible with the pooling size (1080 / 10 = 108)
    # if r.size % pool_size != 0:
    #     raise ValueError(
    #         f"Array size ({r.size}) must be exactly divisible by pool_size ({pool_size})."
    #     )

    # # 2. Reshape the 1D array into a 2D matrix of (Num_Pools, Pool_Size)
    # # For a 1080-element array and pool_size=10, this creates a (108, 10) matrix.
    # num_pools = r.size // pool_size
    # r_reshaped = r.reshape(num_pools, pool_size)
    

    # # 3. Take the minimum along the axis corresponding to the pool size (axis=1)
    # # This reduces the 10 columns into a single column (the minimum value).
    # r = np.min(r_reshaped, axis=1).copy()


    rmin = float(getattr(scan, "range_min", 0.0))
    rmax = float(getattr(scan, "range_max", cap))
    r[~np.isfinite(r)] = np.nan
    r[(r < rmin) | (r > rmax)] = np.nan
    idx = np.arange(r.size, dtype=np.float32)
    valid = np.isfinite(r)

    

    if np.count_nonzero(valid) >= 2:
        r_interp = np.interp(idx, idx[valid], r[valid])
    else:
        r_interp = np.full_like(r, np.nan, dtype=np.float32)

    return r_interp.astype(np.float32)


def _resample_to_n(arr: np.ndarray, n: int) -> np.ndarray:
    m = len(arr)
    if m == n:
        return arr.copy().astype(np.float32)
    if m < 2:
        return np.full((n,), np.nan, dtype=np.float32)
    idx = np.linspace(0, m - 1, n, dtype=np.float32)
    i0 = np.floor(idx).astype(int)
    i1 = np.clip(i0 + 1, 0, m - 1)
    w = idx - i0
    out = (1.0 - w) * arr[i0] + w * arr[i1]
    return out.astype(np.float32)

def downsample_lidar_min(scan: LaserScan, cap: float) -> np.ndarray:
    """
    Downsamples a 1D Lidar range array by taking the minimum of every N consecutive values.

    :param r: The 1080-element Lidar range array (must be divisible by pool_size).
    :param pool_size: The number of elements to pool together (e.g., 10).
    :return: The downsampled array (e.g., 108 elements).
    """

    r = np.asarray(scan.ranges, dtype=np.float32)

    L = len(r)
    
    K = L // 2 # Integer division for 1/4th, which is 270
    # 3. Calculate the pivot index where the cut is made
    # pivot_index = L - K # 1080 - 270 = 810
    # 4. Perform the rotation using slicing
    #    - original_array[pivot_index:]  -> The last K elements (the new front)
    #    - original_array[:pivot_index]  -> The first L-K elements (the new back)
    new_front = r[K:]  # Shape (270,)
    new_back = r[:K]
    rotated_ranges = np.concatenate((new_front, new_back))


    r = rotated_ranges.copy()
    pool_size = 10

    # 1. Ensure the array is compatible with the pooling size (1080 / 10 = 108)
    if r.size % pool_size != 0:
        raise ValueError(
            f"Array size ({r.size}) must be exactly divisible by pool_size ({pool_size})."
        )

    # 2. Reshape the 1D array into a 2D matrix of (Num_Pools, Pool_Size)
    # For a 1080-element array and pool_size=10, this creates a (108, 10) matrix.
    num_pools = r.size // pool_size
    r_reshaped = r.reshape(num_pools, pool_size)
    

    # 3. Take the minimum along the axis corresponding to the pool size (axis=1)
    # This reduces the 10 columns into a single column (the minimum value).
    r_downsampled = np.min(r_reshaped, axis=1)

    return r_downsampled.astype(np.float32)


def deg2rad(d):
    return d * np.pi / 180.0


def sanitize_keep_nan(ranges, rmin, rmax, cap):
    """
    Valid readings clipped to [rmin, min(rmax, cap)].
    Invalids remain np.nan so we can detect/repair them.
    """
    arr = np.asarray(ranges, dtype=np.float32)
    valid = np.isfinite(arr) & (arr >= rmin) & (arr <= rmax)
    out = np.full_like(arr, np.nan, dtype=np.float32)
    if valid.any():
        out[valid] = np.clip(arr[valid], rmin, min(rmax, cap))
    return out


def collect_adjacent_values(vals, i, per_side):
    """
    Collect up to 'per_side' valid values on each side of index i (circular).
    Search nearest-first on each side (i-1, i-2, ..., i+1, i+2, ...).
    Returns a Python list of collected neighbor values (0..2*per_side elements).
    """
    n = vals.size
    collected = []

    # Left side (i-1, i-2, ...)
    found = 0
    for step in range(1, n):  # bounded by n-1 anyway
        idx = (i - step) % n
        if not np.isnan(vals[idx]):
            collected.append(float(vals[idx]))
            found += 1
            if found >= per_side:
                break

    # Right side (i+1, i+2, ...)
    found = 0
    for step in range(1, n):
        idx = (i + step) % n
        if not np.isnan(vals[idx]):
            collected.append(float(vals[idx]))
            found += 1
            if found >= per_side:
                break

    return collected



def fill_with_adjacent_mean_k(vals, rcap, prev_vals=None, neighbors_total=4):
    """
    Fix NaNs by averaging up to 'neighbors_total' adjacent valid rays around each hole (circular).
    neighbors_total is split evenly: per_side = neighbors_total // 2.
    If some neighbors are missing, average whatever is available.
    If none found, try prev_vals[i]; else set to rcap.

    Returns: repaired array and stats dict.
    """
    n = vals.size
    out = vals.copy()
    per_side = max(0, int(neighbors_total) // 2)

    fixed_avg = 0
    fixed_prev = 0
    fixed_cap = 0

    nan_idx = np.where(np.isnan(out))[0]
    for i in nan_idx:
        neighbors = collect_adjacent_values(out, i, per_side) if per_side > 0 else []
        if len(neighbors) > 0:
            out[i] = float(np.mean(neighbors))
            fixed_avg += 1
        else:
            if prev_vals is not None and not np.isnan(prev_vals[i]):
                out[i] = prev_vals[i]
                fixed_prev += 1
            else:
                out[i] = rcap
                fixed_cap += 1

    stats = dict(
        avg=fixed_avg,
        prev=fixed_prev,
        cap=fixed_cap,
        total=int(nan_idx.size),
        neighbors_total=int(neighbors_total),
        per_side=per_side
    )
    return out, stats



def polar_to_cartesian(angles, distances):
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    return np.column_stack((x, y))

def estimate_orientation_pca(points):
    if len(points) < 10:
        return None

    centroid = np.mean(points, axis=0)
    centered = points - centroid

    _, _, vh = np.linalg.svd(centered)
    direction = vh[0]  # main axis

    angle = np.arctan2(direction[1], direction[0])
    angle = wrap_to_pi(angle)

    if angle > np.pi/2:
        angle -= np.pi
    elif angle < -np.pi/2:
        angle += np.pi
    
    return angle



counter = 0
scanner_glob = 0

# ───────────────────────── Plotter thread ─────────────────────────

class LivePlotter:
    """
    Threaded matplotlib plotter: polar 108-ray scan + (v,w) point.
    """
    def __init__(self, rmax_plot: float, limits: ControlLimits, rotate_deg: float = 0.0, refresh_hz: float = 10.0):
        if matplotlib is None or plt is None:
            raise RuntimeError("matplotlib is not available; install it in your venv to enable plotting.")
        self.rmax_plot = float(rmax_plot)
        self.limits = limits
        self.rotate = np.deg2rad(float(rotate_deg))
        self.refresh_dt = 1.0 / float(refresh_hz)

        self._lock = threading.Lock()
        self._lidar108 = np.zeros(108, dtype=np.float32)
        self._v = 0.0
        self._w = 0.0
        self._running = False
        self._thr = None

        # Precompute angles for 108 rays over [-pi, pi)
        self._theta = np.linspace(-np.pi, np.pi, 108, endpoint=False, dtype=np.float32)

        # Prepare figure
        self.fig = plt.figure(figsize=(9, 4.5))
        self.ax_polar = self.fig.add_subplot(1, 2, 1, projection='polar')
        self.ax_cart = self.fig.add_subplot(1, 2, 2)

        # Polar setup
        self.ax_polar.set_theta_zero_location('E')  # 0 at +x
        self.ax_polar.set_theta_direction(-1)       # CW positive to match typical LiDAR orientation
        self.ax_polar.set_rlim(0.0, self.rmax_plot)
        self.ax_polar.set_title("LiDAR (108 rays)")

        # Init line
        self.polar_line, = self.ax_polar.plot(self._theta, np.zeros_like(self._theta))

        # Cartesian v-w setup
        self.ax_cart.set_xlim(-0.05, self.limits.max_lin * 1.05)
        self.ax_cart.set_ylim(-self.limits.max_ang * 1.05, self.limits.max_ang * 1.05)
        self.ax_cart.set_xlabel("v (m/s)")
        self.ax_cart.set_ylabel("w (rad/s)")
        self.ax_cart.set_title("Command (v, w)")
        # Axes
        self.ax_cart.axvline(0.0, lw=1, alpha=0.5)
        self.ax_cart.axhline(0.0, lw=1, alpha=0.5)
        self.vw_point, = self.ax_cart.plot([0.0], [0.0], marker='o')

        self.fig.tight_layout()
        plt.ion()
        self.fig.show()

    def update_data(self, lidar108: np.ndarray, v: float, w: float):
        with self._lock:
            if lidar108 is not None and lidar108.size == 108:
                self._lidar108 = lidar108.astype(np.float32)
            self._v = float(v)
            self._w = float(w)

    def _tick(self):
        with self._lock:
            r = np.clip(self._lidar108, 0.0, self.rmax_plot)
            theta = (self._theta + self.rotate).astype(np.float32)
            v, w = self._v, self._w

        # Update polar
        self.polar_line.set_data(theta, r)
        # Update v-w point
        self.vw_point.set_data([v], [w])

        # Redraw
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        self._running = False
        if self._thr is not None:
            self._thr.join(timeout=1.0)

    def _loop(self):
        while self._running:
            try:
                self._tick()
            except Exception:
                pass
            time.sleep(self.refresh_dt)


# ───────────────────────── Main runner ─────────────────────────

class TB4PolicyRunner(Node):
    def __init__(
        self,
        model_path: str,
        vecnorm_path: Optional[str],
        trainer_hint: Optional[str],
        goals: GoalParams,
        lidar_params: LidarParams,
        limits: ControlLimits,
        num_goals: int = 2,
        current_goal: int = 0,
        rate_hz: float = 4.0,
        obs_dim: int = 182,
        deterministic: bool = True,
        stop_on_success: bool = True,
        enable_plot: bool = False,
        rmax_plot: float = 7.0,
        rotate_deg: float = 0.0,
        plot_hz: float = 10.0,
        base_x: float = 0.0,
        base_y: float = 0.0,
        base_yaw: float = 0.0,
        find_orientation: bool = True
       
    ):
        super().__init__("tb4_run_policy")
        self.get_logger().info(f"Loading policy from: {model_path}")
        self.model = _load_sb3_model(model_path, trainer_hint, device="cpu")
        self.det = deterministic

        # VecNormalize stats
        self.vecnorm = None
        if vecnorm_path and os.path.isfile(vecnorm_path):
            self.get_logger().info(f"Loading VecNormalize stats from: {vecnorm_path}")
            dummy_vec = DummyVecEnv([lambda: DummyObsEnv(obs_dim=obs_dim)])
            self.vecnorm = VecNormalize.load(vecnorm_path, dummy_vec)
            self.vecnorm.training = False
            self.vecnorm.norm_reward = False
        else:
            if vecnorm_path:
                self.get_logger().warn(f"VecNormalize file not found: {vecnorm_path} (continuing without normalization)")
            else:
                self.get_logger().info("No VecNormalize path provided — continuing without normalization.")

        # Params/state v, w: (0.25, -1.0)

        self.goals = goals
        self.num_goals = len(goals)-1 
        self.current_goal = 0
        self.lp = lidar_params
        self.limits = limits
        self.rate_hz = float(rate_hz)
        self.dt = 1.0 / self.rate_hz
        self.stop_on_success = stop_on_success

        self.global_x = base_x
        self.global_y = base_y
        self.global_yaw = base_yaw
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # Offset variables:
        self.target_count = 10
        self.angle_count = 0
        self.offset_sum = 0
        self.prev_vector = None
        self.find_orientation = find_orientation
        self.start_saving = False

        # Store data
        self.store_pos = []
        self.store_angles = [] # shape to be the number of angles that are used, 540 or 1080
        self.store_ranges = [] # shape to be (number of angles, number of steps)
        self.store_time = []          # relative seconds
        self.store_time_ns = []       # absolute ROS nanoseconds
        self.start_time_ns = None
        self.store_target_cmd = []
        self.store_filtered_cmd = []
        self.store_odom_cmd = []
                
        # ROS I/O
        self.cmd_pub = self.create_publisher(Twist, "/turtlebot1/cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "/turtlebot1/scan", self._on_scan, qos_profile_sensor_data)
        odom_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST)
        self.odom_sub = self.create_subscription(Odometry, "/turtlebot1/odom", self._on_odom, odom_qos)
        self.timer = self.create_timer(self.dt, self._on_timer)

        # Buffers
        self.last_scan: Optional[LaserScan] = None
        self.last_odom: Optional[Odometry] = None
        self.pose_xy: Optional[Tuple[float, float]] = None
        self.yaw: Optional[float] = None
        self.prev_feats = None

        # Plotter
        self.plotter: Optional[LivePlotter] = None
        self.enable_plot = bool(enable_plot)
        self.rmax_plot = float(rmax_plot)
        self.rotate_deg = float(rotate_deg)
        self.plot_hz = float(plot_hz)

        if self.enable_plot:
            if matplotlib is None or plt is None:
                self.get_logger().warn("Matplotlib not available; plotting disabled.")
            else:
                self.plotter = LivePlotter(self.rmax_plot, self.limits, rotate_deg=self.rotate_deg, refresh_hz=self.plot_hz)
                self.plotter.start()

        self.get_logger().info("TB4PolicyRunner initialized. Waiting for /scan and /odom…")

    # ───────────────────────── ROS Callbacks ─────────────────────────
    def _on_scan(self, msg: LaserScan):
        if self.last_scan is None:
            self.get_logger().info("First /scan received")
        self.last_scan = msg

        if self.last_odom is None:
            return

        msg_odom = self.last_odom
        p = msg_odom.pose.pose.position
        q = msg_odom.pose.pose.orientation

        yaw_local = yaw_from_quat(q.x, q.y, q.z, q.w, 0.0)
        # 2. Transform position (ROTATION + TRANSLATION)
        x_global, y_global = translate_position(
            p.x, p.y,
            self.global_x, self.global_y,
            self.global_yaw
        )
        yaw_global = wrap_to_pi(yaw_local + self.global_yaw)
        self.pose_xy = (x_global, y_global)
        # self.pose_xy = (p.x, p.y)
        self.yaw = yaw_global

        # -------------------------------------------------
    # STORE DATA HERE
    # -------------------------------------------------

        if self.start_saving:

            #
            # Use SCAN timestamp
            #
            stamp = msg.header.stamp

            #
            # Convert ROS time -> nanoseconds
            #
            t_ns = stamp.sec * 1_000_000_000 + stamp.nanosec

            #
            # Initialize experiment start time
            #
            if self.start_time_ns is None:
                self.start_time_ns = t_ns

            #
            # Relative time in seconds
            #
            t_rel = (t_ns - self.start_time_ns) * 1e-9

            #
            # Store timestamps
            #
            self.store_time.append(t_rel)
            self.store_time_ns.append(t_ns)

            #
            # Store pose
            #
            self.store_pos.append([
                self.pose_xy[0],
                self.pose_xy[1],
                self.yaw
            ])

            #
            # Store ranges
            #
            self.store_ranges.append(
                np.array(msg.ranges, dtype=np.float32)
            )



        # self.yaw = yaw_local

        # print(f'p.x, p.y = {p.x}, {p.y}, yaw_local = {yaw_local}')
        # print(f"x_global, y_global = {x_global}, {y_global}, yaw_global = {yaw_global}")
        # print()




    def _on_odom(self, msg: Odometry):
        if self.last_odom is None:
            self.get_logger().info("First /odom received")
        
        self.last_odom = msg
        

        # p = msg.pose.pose.position
        # q = msg.pose.pose.orientation

        # yaw_local = yaw_from_quat(q.x, q.y, q.z, q.w, 0.0)
        # # 2. Transform position (ROTATION + TRANSLATION)
        # x_global, y_global = translate_position(
        #     p.x, p.y,
        #     self.global_x, self.global_y,
        #     self.global_yaw
        # )
        # yaw_global = wrap_to_pi(yaw_local + self.global_yaw)
        # # self.pose_xy = (x_global, y_global)
        # self.pose_xy = (p.x, p.y)
        # # self.yaw = yaw_global
        # self.yaw = yaw_local

        # print(f'p.x, p.y = {p.x}, {p.y}, yaw_local = {yaw_local}')
        # print(f"x_global, y_global = {x_global}, {y_global}, yaw_global = {yaw_global}")
        # print()
    
    def _laser_cb(self):
        # Angles for every incoming ray
        # print('HERE')
        msg = self.last_scan
        n = len(msg.ranges)
        rotate_rad = np.deg2rad(90)
        rcap = 4.0
        neighbors_total = 4
        angles = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment
        angles = wrap_to_pi(angles + rotate_rad)
        if self.start_saving:
            self.store_angles = angles

        # Sanitize, keep NaNs for invalids
        d_raw = sanitize_keep_nan(msg.ranges, msg.range_min, msg.range_max, cap=rcap)
       
        d_fixed, stats = fill_with_adjacent_mean_k(
            d_raw, rcap, prev_vals=self.prev_vector, neighbors_total=neighbors_total
        )

        points = polar_to_cartesian(angles, d_fixed)
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        

        # remove too close noisew
        mask = (np.linalg.norm(points, axis=1) > 0.1)
        points = points[mask]

   
        corridor_angle = estimate_orientation_pca(points)
        # print('corridor angle:', corridor_angle )
        self.prev_vector = d_fixed
        return corridor_angle

    # ───────────────────────── Helpers ─────────────────────────
    def _compute_goal_polar(self) -> Optional[Tuple[float, float]]:
        goal_x = self.goals[self.current_goal].x
        goal_y = self.goals[self.current_goal].y
        if self.pose_xy is None or self.yaw is None:
            return None
        
        dx = goal_x - self.pose_xy[0]
        dy = goal_y - self.pose_xy[1]
        distance = math.hypot(dx, dy)
        # print(f'self x, y, yaw = {self.pose_xy}, {self.yaw}, in deg = {np.rad2deg(self.yaw)}')

        print(f'pos: {self.pose_xy[0]:.2f}, {self.pose_xy[1]:.2f}, goal: {goal_x}, {goal_y}')
        tgt_theta = math.atan2(dy, dx)
        rel = (tgt_theta - self.yaw + math.pi) % (2 * math.pi) - math.pi
        return float(distance), float(rel)

    def _lidar_to_108(self, scan: LaserScan) -> np.ndarray:
        arr = _sanitize_ranges(scan, cap=float(self.lp.lidar_max))
        arr108 = _resample_to_n(arr, self.lp.num_rays)

        arr108 = np.where(np.isfinite(arr108), arr108, self.lp.lidar_max).astype(np.float32)

        return np.clip(arr108, 0.0, float(self.lp.lidar_max))


    # def _store_data(self, to_save = False):
        # pose_x = self.pose_xy[0]
        # pose_y = self.pose_xy[1]
        # pose_theta = self.yaw
        # if self.start_saving:
        #     self.store_pos.append([self.pose_xy[0], self.pose_xy[1], self.yaw])
        #     self.store_ranges.append(self.last_scan.ranges)

        # if to_save:
        #     Pose = np.array(self.store_pos)
        #     angles = np.array(self.store_angles)
        #     ranges = np.array(self.store_ranges).T
            

        #     np.savez("data.npz", Pose=Pose, angles=angles, ranges=ranges)

    def _store_data(self, to_save=False):

        if to_save:

            Pose = np.array(self.store_pos)
            angles = np.array(self.store_angles)
            ranges = np.array(self.store_ranges).T
            times = np.array(self.store_time)
            times_ns = np.array(self.store_time_ns)
            target_cmd = np.array(self.store_target_cmd)
            filtered_cmd = np.array(self.store_filtered_cmd)
            odom_cmd = np.array(self.store_odom_cmd)

            np.savez(
                "data.npz",
                Pose=Pose,
                angles=angles,
                ranges=ranges,
                time=times,
                time_ns=times_ns,
                target_cmd=target_cmd,
                filtered_cmd=filtered_cmd,
                odom_cmd=odom_cmd
            )

            self.get_logger().info(
                f"Saved dataset with {len(times)} scans"
            )
            

 

    # ───────────────────────── Control loop ─────────────────────────
    def _build_observation(self) -> Optional[np.ndarray]:
        """
        Here, the _lidar_to_features has to be changed. There the k_bins are used.
        """
        if self.last_scan is None:
            return None
        
        if self.angle_count < self.target_count and self.find_orientation:
            offset_angle = self._laser_cb()
            self.offset_sum += offset_angle
            self.angle_count += 1
            return None
        
        if self.angle_count == self.target_count and self.find_orientation:
            yaw_offset = self.offset_sum/self.target_count
            self.global_yaw = math.pi/2 - yaw_offset
            self.base_yaw = self.global_yaw
            self.angle_count += 1
            self.get_logger().info(f"Orientation Found, yaw_offset = {yaw_offset} in deg:{np.rad2deg(yaw_offset)} base yaw:{self.base_yaw}, in deg:{np.rad2deg(self.base_yaw)}")
            self.start_saving = True
            # self._store_data(to_save = True)
            # raise SystemExit
        self._store_data(to_save = False)
        # print(f"global_yaw: {self.global_yaw}, offset_sum = {self.offset_sum}")
        # print(f"global_yaw = {self.global_yaw}")
        goal = self._compute_goal_polar()
        if goal is None:
            return None

        dist, rel = goal
        
        lidar_108 = self._lidar_to_108(self.last_scan)

        obs = np.concatenate([
            lidar_108,  
            np.array([dist, rel], dtype=np.float32)
        ]).astype(np.float32)


        # update plot data
        if self.plotter is not None:
            self.plotter.update_data(lidar_108, getattr(self, "_last_v", 0.0), getattr(self, "_last_w", 0.0))

        return obs

    # def _on_timer(self):
    #     obs = self._build_observation()
    #     if obs is None:
    #         return
    #     dist = float(obs[len(obs) - 2])
    #     # print(f'base_yaw = {np.rad2deg(self.base_yaw)} deg, global_yaw = {np.rad2deg(self.global_yaw)}, deg')
    #     # print()
        

    #     # Check if we reached the current goal
    #     if dist <= self.goals[self.current_goal].success_radius:
    #         if self.current_goal == len(self.goals) - 1:
    #             offset_angle = self._laser_cb()
    #             # self._store_data(to_save=True)
    #             self._publish_cmd(0.0, 0.0)
    #             self.get_logger().info(f"✅ FINAL Goal Reached in {self.timer} seconds! Radius: {self.goals[self.current_goal].success_radius}")
    #             if self.stop_on_success:
    #                 if self.plotter is not None:
    #                     self.plotter.stop()
    #                 raise SystemExit
    #             return
    #         else:
    #             self.get_logger().info(f"✅ Waypoint {self.current_goal + 1} reached. Moving to next.")
    #             self.current_goal += 1

    #     nobs = self._normalize_obs_if_needed(obs)

    #     action, _ = self.model.predict(nobs, deterministic=self.det)

    #     throttle = float(np.clip(action[0], 0.0, 1.0))
    #     steer = float(np.clip(action[1], -1.0, 1.0))
  

    #     v = self.limits.max_lin * throttle
    #     w = self.limits.max_ang * steer

    #     # gentle distance scaling
    #     v *= float(np.clip(dist / 1.0, 0.2, 1.0))


    #     self._publish_cmd(v, w)
    #     # self._publish_cmd(0,0)

    def _on_timer(self):

        obs = self._build_observation()

        if obs is None:
            return

        dist = float(obs[-2])

        # Goal reached logic
        if dist <= self.goals[self.current_goal].success_radius:

            # Final goal
            if self.current_goal == len(self.goals) - 1:
                offset_angle = self._laser_cb()

                # Reset filtered velocities
                self.linear_velocity = 0.0
                self.angular_velocity = 0.0

                # Stop robot
                self._publish_cmd(0.0, 0.0)

                self.get_logger().info(
                    f"✅ FINAL Goal Reached! "
                    f"Radius: {self.goals[self.current_goal].success_radius}"
                )

                if self.stop_on_success:

                    if self.plotter is not None:
                        self.plotter.stop()

                    raise SystemExit

                return

            # Intermediate waypoint
            else:

                self.get_logger().info(
                    f"✅ Waypoint {self.current_goal + 1} reached. Moving to next."
                )

                self.current_goal += 1

                return

        # Normalize observation
        nobs = self._normalize_obs_if_needed(obs)

        # Policy inference
        action, _ = self.model.predict(
            nobs,
            deterministic=self.det
        )

        # Policy outputs
        throttle = float(np.clip(action[0], 0.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))

        # Target velocities
        target_v = self.limits.max_lin * throttle
        target_w = self.limits.max_ang * steer

        # Distance-based slowing near goal
        target_v *= float(
            np.clip(dist / 1.0, 0.2, 1.0)
        )

        # Time step
        dt = self.dt

        # Velocity filter time constants
        tau_v = 0.3
        tau_w = 0.3

        # Convert to discrete smoothing coefficients
        alpha_v = dt / (tau_v + dt)
        alpha_w = dt / (tau_w + dt)

        # First-order velocity dynamics
        self.linear_velocity += alpha_v * (
            target_v - self.linear_velocity
        )

        self.angular_velocity += alpha_w * (
            target_w - self.angular_velocity
        )

        # Coupled angular velocity constraint
        allowed_w = self.limits.max_ang * (
            1.0 - abs(self.linear_velocity) / self.limits.max_lin
        )

        allowed_w = max(0.0, allowed_w)

        self.angular_velocity = np.clip(
            self.angular_velocity,
            -allowed_w,
            allowed_w
        )

        # Coupled linear velocity constraint
        allowed_v = self.limits.max_lin * (
            1.0 - abs(self.angular_velocity) / self.limits.max_ang
        )

        allowed_v = max(0.0, allowed_v)

        self.linear_velocity = np.clip(
            self.linear_velocity,
            0.0,
            allowed_v
        )

        if self.start_saving:
            # Raw policy targets
            self.store_target_cmd.append([
                target_v,
                target_w
            ])

            # Filtered/published velocities
            self.store_filtered_cmd.append([
                self.linear_velocity,
                self.angular_velocity
            ])

            
            last_odom = self.last_odom
            lin_odom = last_odom.twist.twist.linear.x
            ang_odom = last_odom.twist.twist.angular.z

            self.store_odom_cmd.append([
                lin_odom,
                ang_odom
            ])

        # Publish smoothed commands
        self._publish_cmd(
            self.linear_velocity,
            self.angular_velocity
        )

    def _normalize_obs_if_needed(self, obs: np.ndarray) -> np.ndarray:
        if self.vecnorm is None:
            return obs
        obs_b = obs[None, :]
        nobs = self.vecnorm.normalize_obs(obs_b)
        return np.asarray(nobs[0], dtype=np.float32)

    def _publish_cmd(self, v: float, w: float):
        self._last_v = float(np.clip(v, -self.limits.max_lin, self.limits.max_lin))
        self._last_w = float(np.clip(w, -self.limits.max_ang, self.limits.max_ang))
        
        msg = Twist()
        msg.linear.x = self._last_v
        msg.angular.z = self._last_w
        self.cmd_pub.publish(msg)
    
    def reset_pose(self):
        client = self.create_client(ResetPose, "/turtlebot1/reset_pose")

        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for reset_pose service...")

        request = ResetPose.Request()

        future = client.call_async(request)

        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info("Robot pose reset successfully")
        else:
            self.get_logger().error("Failed to reset robot pose")

    def destroy_node(self):
        try:
            if self.plotter is not None:
                self.plotter.stop()
        finally:
            super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="Run SB3 policy on TurtleBot4 to reach a goal (x, y) + live plots")
    parser.add_argument("--model", default="Test_Linux_noise5M.zip", required=True, help="Path to trained policy .zip")
    parser.add_argument("--vecnorm", default='Test_Linux_noise5M.pkl', help="Path to VecNormalize .pkl")
    parser.add_argument("--trainer", default='TQC', help="Optional hint (PPO/SAC/TD3/TQC/A2C/DQN)")
    parser.add_argument("--rate", type=float, default=4.0, help="Control rate (Hz)")
    parser.add_argument("--num_rays", type=int, default=180, help="LiDAR rays expected by the policy")
    parser.add_argument("--lidar_max", type=float, default=30.0)
    parser.add_argument("--inv_alpha", type=float, default=0.6)
    parser.add_argument("--dist_floor", type=float, default=0.05)
    parser.add_argument("--k_bins", type=int, default=24)
    parser.add_argument("--short_r", type=float, default=1.0)
    parser.add_argument("--max_lin", type=float, default=0.4)
    parser.add_argument("--max_ang", type=float, default=1.0)
    parser.add_argument("--success_radius", type=float, default=0.25)
    parser.add_argument("--no_stop", action="store_true", help="Do not shutdown when goal reached")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic=True for predict()")
    parser.add_argument("--find_orientation", type=bool, default=True, help="Apply PCA transform for finding the correct orientation")

    # Plotting options
    parser.add_argument("--plot", action="store_true", help="Enable live plotting windows")
    parser.add_argument("--rmax_plot", type=float, default=4.0, help="Radar plot max range (m)")
    parser.add_argument("--rotate_deg", type=float, default=0.0, help="Rotate polar plot (deg, CCW positive)")
    parser.add_argument("--plot_hz", type=float, default=10.0, help="Plot refresh rate (Hz)")

    args = parser.parse_args()

    rclpy.init()

    final_success_radius = 0.25
    checkpoint_success_radius = 0.8

    # base_x=17.62
    # base_y=20.306
    goals = [[17.6, 22.3],
             [14.956, 22.618],
             [1.503, 22.633],
             [-5.893, 22.991],
             [-6.873, 19.245],
             [-6.873, 8.458],
             [-6.584, 0.597],
             [-5.107, -6.143],
             [4.02, -8.118],
             [11.00, -8.134],
             [17.511, -2.1],
             [17.593, 4.008],
             [17.593, 12.719],
             [17.617, 20.306],
             [17.6, 22.3], # start again
             [14.956, 22.618],
             [1.503, 22.633],
             [-5.893, 22.991],
             [-6.873, 19.245],
             [-6.873, 8.458],
             [-7.162, 3.347],
             [-6.584, 0.597],
             [-5.107, -6.143],
             [4.02, -8.118],
             [11.00, -8.134],
             [17.511, -2.1],
             [17.593, 4.008],
             [17.593, 12.719],
             [17.617, 20.306],
            ]
    # goals = [[17.6, 22.3],
    #          [14.956, 22.618],
    #         #  [8.535, 22.633],
    #          [1.503, 22.633],
    #          [-5.893, 22.991],
    #          [-6.873, 19.245],
    #          [-6.873, 8.458],
    #         #  [-7.162, 3.347],
    #          [-6.584, 0.597],
    #         #  [-7.019, -2.881],
    #         #  [-6.838, -5.265],
    #          [-5.107, -6.143],
    #          [4.02, -8.118],
    #          [11.00, -8.134],
    #         #  [16.838, -6.525],
    #          [17.511, -2.1],
    #          [17.593, 4.008],
    #          [17.593, 12.719],
    #          [17.617, 20.306],
    #          [17.6, 22.3], # start again
    #          [14.956, 22.618],
    #         #  [8.535, 22.633],
    #          [1.503, 22.633],
    #          [-5.893, 22.991],
    #          [-6.873, 19.245],
    #          [-6.873, 8.458],
    #         #  [-7.162, 3.347],
    #         #  [-6.584, 0.597],
    #         # #  [-7.019, -2.881],
    #         # #  [-6.838, -5.265],
    #         #  [-5.107, -6.143],
    #         #  [4.02, -8.118],
    #         #  [11.00, -8.134],
    #         # #  [16.838, -6.525],
    #         #  [17.511, -2.1],
    #         #  [17.593, 4.008],
    #         #  [17.593, 12.719],
    #         #  [17.617, 20.306],
    #         ]
    # base_x=17.62
    # base_y=20.306
    # goals =[[-2,0]]
   
    num_points = len(goals)
    counter = 0
    goal_params = []
    for g in goals:
        if counter < num_points - 1:
            checkpoint = GoalParams(x=g[0], y=g[1], success_radius=checkpoint_success_radius)
            goal_params.append(checkpoint)

        else:
            final_goal = GoalParams(x=g[0], y=g[1], success_radius=final_success_radius)
            goal_params.append(final_goal)
        counter += 1

    

    # goal = GoalParams(x=args.target_x, y=args.target_y, success_radius=args.success_radius)
    lp = LidarParams(
        num_rays=args.num_rays,
        lidar_max=args.lidar_max,
        inv_alpha=args.inv_alpha,
        dist_floor=args.dist_floor,
        k_bins=args.k_bins,
        short_r=args.short_r,
    )
    limits = ControlLimits(max_lin=args.max_lin, max_ang=args.max_ang)

    node = TB4PolicyRunner(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        trainer_hint=args.trainer,
        goals=goal_params,
        num_goals=len(goal_params),
        lidar_params=lp,
        limits=limits,
        rate_hz=args.rate,
        obs_dim = 182,
        deterministic=args.deterministic,
        stop_on_success=not args.no_stop,
        enable_plot=args.plot,
        rmax_plot=args.rmax_plot,
        rotate_deg=args.rotate_deg,
        plot_hz=args.plot_hz,
        find_orientation = args.find_orientation,
        base_x=17.62,
        base_y=20.306,
        # base_x=0,
        # base_y=0,
        base_yaw = 0
    )
    # ако сме по дължина на коридора
    # base_yaw = -math.pi/2 - (chisloto v radiani ot hough transforma)
    
    # RESET THE POSE
    if args.find_orientation:
        node.reset_pose()
    
    start = time.time()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        # This catch block handles both Ctrl+C and our auto-shutdown
        node._store_data(to_save=True)
        stop = time.time()
        node.get_logger().info(f"Shutting down node.... Time: {stop - start} seconds")
    finally:
        # One last safety check: ensure velocity is zero
        node._publish_cmd(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
    # run command:
    # python3 tb4_run_tqc_noise.py --model='Test_Linux_noise5M.zip' --vecnorm='Test_Linux_noise5M.pkl' --trainer TQC --target_x 2.43 --target_y 1.88

# python3 tb4_run_tqc_noise.py --model='two_NN_180r_20M.zip' --vecnorm='two_NN_180r_20M.pkl' --trainer TQC --target_x -3.0 --target_y 3.0

# command to run for waypoints:
# python3 tb4_run_tqc_front_lasers.py --model='two_NN_face_all_cases_50M.zip' --vecnorm='two_NN_face_all_cases_50M.pkl' --trainer TQC