import os
import time
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from contourpy.util.data import simple
from gymnasium import spaces
import mujoco
import mujoco.viewer
import heapq

print_one = 0
class CubeSimulationEnv_class(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, num_rays=0, render_mode=None, model_path="xml_models/simulation_model.xml",
                 training_mode=True):
        super().__init__()

        self.num_rays = num_rays
        self.training_mode = training_mode
        self.max_episode_steps = 1000
        self.current_step = 0
        self.render_mode = render_mode
        self.model_path = model_path
        self.previous_distance = 30
        self.episode_return = 0
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.last_episode_result = None

        self.episode_time_begin = 0
        self.relative_azimuth = 0

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        """------------WAYPOINTS----------"""
        self.waypoints = []
        self.current_waypoint_idx = 0

        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32)
        """OBSERVATION SPACE WITH LIDAR SCANS"""
        self.observation_space = spaces.Box(
            low=np.array([0.0] * num_rays + [0.0, -np.pi]),
            high=np.array([30.0] * num_rays + [30.0, np.pi]),
            dtype=np.float32
        )
        # Load MuJoCo model
        self.xml_model = self._load_and_modify_xml_model()

        # Load model with asset directory set
        self.model = mujoco.MjModel.from_xml_string(self.xml_model)
        self.data = mujoco.MjData(self.model)

        # Get cube body ID
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_body")

        # Get sensor IDs for the LiDAR rays
        self.lidar_sensor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            for i in range(self.num_rays)
        ]

        # Viewer
        self.viewer = None
        if self.render_mode == "human":
            self._setup_viewer()

        # Reset the environment
        self.reset()




    def _load_and_modify_xml_model(self):
        """Load and modify the XML model with proper asset path resolution."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found.")

        tree = ET.parse(self.model_path)
        root = tree.getroot()

        # Get the project root directory
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_file_dir)
        assets_dir = os.path.join(project_root, "assets")

        # Fix asset paths (same as before)
        asset_element = root.find("asset")
        if asset_element is not None:
            for mesh in asset_element.findall("mesh"):
                file_attr = mesh.get("file")
                if file_attr:
                    filename = os.path.basename(file_attr)
                    correct_path = os.path.join(assets_dir, filename)
                    if os.path.exists(correct_path):
                        mesh.set("file", correct_path)
                        print(f"✅ Updated mesh path: {correct_path}")

        cube_body = None
        for body in root.findall(".//body"):
            if body.get("name") == "cube_body":
                cube_body = body
                break

        if cube_body is None:
            raise ValueError("Cube body not found in the XML model.")

        """UNCOMMENT THIS PART WHEN YOU WANT THE LIDARS"""

        sensor = None
        for s in root.findall(".//sensor"):
            sensor = s
            break

        if sensor is None:
            raise ValueError("Sensor element not found in the XML model.")

        for i in range(self.num_rays):
            # Evenly distribute rays around 360°, starting from forward (0°)
            # angle = 2 * np.pi * i / self.num_rays

            # Make the rays 180°facing front
            # here "+ np.pi*3/2" moves the scans starting from 270 deg to 90 deg (CCW)
            angle = np.pi * i / self.num_rays + np.pi*3/2

            cos_angle = np.cos(angle)

            sin_angle = np.sin(angle)

            site = ET.SubElement(cube_body, "site")
            site.set("name", f"lidar_site_{i}")
            site.set("pos", f"{-0.05} {-0.05} -0.1")
            site.set("size", "0.05")
            site.set("rgba", "1 0 0 1")
            site.set("zaxis", f"{cos_angle} {sin_angle} 0")  # Point in the ray direction

            rangefinder = ET.Element("rangefinder")
            rangefinder.set("name", f"lidar_{i}")
            rangefinder.set("site", f"lidar_site_{i}")
            sensor.append(rangefinder)

        return ET.tostring(root, encoding="unicode")

    def _setup_viewer(self):

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Set up camera for a top-down view
        self.viewer.cam.distance = 125.0
        self.viewer.cam.azimuth = 0.0
        self.viewer.cam.elevation = -70.0
        self.viewer.cam.lookat[:] = [0, 0, 1]

    def _get_obs(self):
        """
        CHANGE THIS PART(UNCOMMENT THE LIDAR_READINGS WHEN YOU WANT THE LIDARS
        AND CHANGE THE OBSERVATION SPACE
        """
        # Process LiDAR readings
        self.lidar_readings = np.array([self.data.sensordata[lidar_id] for lidar_id in self.lidar_sensor_ids])
        # before_noise = self.lidar_readings.copy()

        self.lidar_readings = np.where(self.lidar_readings == -1, 30, self.lidar_readings)

        """
        Put noise in the lidar readings based on the distance + modify 5% of the readings to be full scale
        """
        noise = 0.02

        self.lidar_readings = self.lidar_readings + np.random.normal(0, noise*self.lidar_readings)
        self.lidar_readings = np.clip(self.lidar_readings, 0, 30)

        # # Get the reading's indices which are not full scale
        # not_full_idx = np.where(self.lidar_readings < 30)[0]
        # indices = np.random.choice(len(not_full_idx), size=int(len(self.lidar_readings)/20), replace=False) # to replace
        # self.lidar_readings[not_full_idx[indices]] = 30



        # Get positions
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_body")
        cube_pos = self.data.xpos[cube_body_id].copy()

        sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        # sphere_pos = self.data.geom_xpos[sphere_geom_id].copy()
        sphere_pos = self.waypoints[self.current_waypoint_idx]
        # Get orientation of the cube (rotation matrix)
        cube_xmat = self.data.xmat[cube_body_id].reshape(3, 3)

        # Cube's forward vector in world frame is the first column of the rotation matrix
        cube_forward = cube_xmat[:, 0]

        # Heading (yaw) angle of cube in global frame (assuming 2D movement on xy-plane)
        cube_yaw = np.arctan2(cube_forward[1], cube_forward[0])

        # Vector from cube to sphere
        relative_pos = sphere_pos - cube_pos
        distance = np.linalg.norm(relative_pos[:2])
        azimuth_global = np.arctan2(relative_pos[1], relative_pos[0])

        # Relative azimuth (angle between agent's heading and the target)
        self.relative_azimuth = azimuth_global - cube_yaw
        self.relative_azimuth = (self.relative_azimuth + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
        """
        TO CHANGE LATER: THE OBSERVATIONS ARE JUST DISTANCE AND ANGLE
        """
        # self.lidar_readings = np.full(108, 5)
        # new_readings = np.full(108, 30)

        # global print_one
        # if print_one == 0:
        #     print(f'lidar reading shape:{self.lidar_readings}')
        #     print_one += 1
        # print(f'New reading shape:{new_readings.shape}')

        # Combine into observation
        # obs = np.array([distance, self.relative_azimuth])
        obs = np.concatenate([self.lidar_readings, [distance, self.relative_azimuth]])
        # obs = np.array([distance, self.relative_azimuth])


        return obs.astype(np.float32)

    def _get_info(self):
        # Get positions using direct geom access instead of sensors
        cube_pos = np.zeros(3)
        sphere_pos = np.zeros(3)

        # Get cube position from its body
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_body")
        if cube_body_id >= 0:
            # Get position of the cube body in world coordinates
            cube_pos = self.data.xpos[cube_body_id].copy()

        # Get sphere position directly from its geom
        sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        if sphere_geom_id >= 0:
            # Get position of the sphere geom in world coordinates
            sphere_pos = self.data.geom_xpos[sphere_geom_id].copy()

        # Calculate distance to sphere
        """
            CHANGED DISTANCE TO SPHERE FROM 3D DISTANCE TO 2D DISTANCE
            BECASUE THE ROBOT CAN'T MOVE IN THE Z DIRECTION
        """
        # distance_to_sphere = np.linalg.norm(sphere_pos[:2] - cube_pos[:2])
        distance_to_sphere = np.linalg.norm(sphere_pos - cube_pos)

        return {
            "distance_to_sphere": distance_to_sphere,
            "cube_position": cube_pos,
            "sphere_position": sphere_pos
        }



    def reset(self, seed=None, options=None):
        # --- (Episode Stats Update Code remains unchanged) ---
        self.episode_time_length = time.time() - self.episode_time_begin
        if self.current_step > 0:
            self.last_episode_info = {"episode_time_length": self.episode_time_length}
        if self.last_episode_result == "success":
            self.success_count += 1
        elif self.last_episode_result == "collision":
            self.collision_count += 1
        elif self.last_episode_result == "timeout":
            self.timeout_count += 1
        epi_count = self.episode_count - 1
        if self.episode_count > 1:
            self.success_rate = self.success_count / (epi_count)
            self.collision_rate = self.collision_count / (epi_count)
            self.timeout_rate = self.timeout_count / (epi_count)
        if self.last_episode_result == "success" and self.training_mode == False:
            print(
                f"SUCCESS: Eval_episode = {epi_count} sr={self.success_rate:.2f}, cr={self.collision_rate:.2f}, tr={self.timeout_rate:.2f}, return={self.episode_return:.2f}")
        elif self.last_episode_result == "collision" and self.training_mode == False:
            print(
                f"COLLISION: Eval_episode = {epi_count} sr={self.success_rate:.2f}, cr={self.collision_rate:.2f}, tr={self.timeout_rate:.2f}, return={self.episode_return:.2f}")
        elif self.last_episode_result == "timeout" and self.training_mode == False:
            print(
                f"TIMEOUT: Eval_episode = {epi_count} sr={self.success_rate:.2f}, cr={self.collision_rate:.2f}, tr={self.timeout_rate:.2f}, return={self.episode_return:.2f}")

        self.episode_count += 1
        self.last_episode_result = None
        self.episode_time_begin = time.time()

        super().reset(seed=seed)

        # Reset step counter and return
        self.current_step = 0
        self.episode_return = 0
        self.previous_distance = 30
        self.stuck_counter = 0

        # Define arena boundaries with buffer

        """The space is 15x15"""
        wall_buffer = 0.2
        min_x = -7.5 + wall_buffer
        max_x = 7.5 - wall_buffer
        min_y = -7.5 + wall_buffer
        max_y = 7.5 - wall_buffer

        min_separation = 3.0

        # 1. Place cube randomly
        # cube_x = np.random.uniform(min_x + 0.32, max_x - 0.32)
        # cube_y = np.random.uniform(min_y + 0.32, max_y - 0.32)
        # cube_yaw = np.random.uniform(-np.pi, np.pi)

        # sphere_x, sphere_y = self._find_valid_sphere_position(
        #     cube_x, cube_y, min_x + 1, max_x - 1, min_y + 1, max_y - 1, min_separation
        # )
        # sphere_x = -20
        # sphere_y = 3.5

        """
                                                                  x
                                                                  ^
        Robot's orientation in the reference frame of mujoco y<---|
        yaw = 0 → pointing along + X
        yaw = +π / 2 → pointing along + Y
        yaw = π → pointing along - X
        yaw = -π / 2 → pointing along - Y 
        """
        # cube_x = 16.06
        # cube_y = 22.66
        cube_x = 17.62
        cube_y = 20.306
        cube_yaw = np.pi

        self.waypoints = [
            np.array([10, 22.5, 2.0]),
            np.array([-7.15, 22.5, 2.0]),
            np.array([-7.05, 14.3, 2.0]),
            np.array([-7.05, 6.15, 2.0]),
            np.array([-6.36, 6.34, 2.0]),
            np.array([-7.35, 2.81, 2.0]),
            np.array([-6.67, -0.34, 2.0]),
            np.array([-6.71, -4.74, 2.0]),
            np.array([-6.6, -6.0, 2.0]),
            np.array([-3.0, -6.6, 2.0]),
            # np.array([13.0, -6.0, 2.0]),
            # np.array([17.53, -5.33, 2.0]),
            # np.array([17.53, 0.5, 2.0]),
            # # np.array([17.53, 3.5, 2.0]),
            # # np.array([17.53, 7.5, 2.0]),
            # np.array([17.53, 13.5, 2.0]),
            # # np.array([17.53, 18.5, 2.0]),
            # np.array([17.5, 20.5, 2.0]),
            # np.array([15.5, 22.8, 2.0]),

        ]


        self.current_waypoint_idx = 0

        sphere_goal_idx = len(self.waypoints) - 1


        # Set cube and sphere positions
        self.data.qpos[:3] = [cube_x, cube_y, cube_yaw]
        sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        self.model.geom_pos[sphere_geom_id, :] = self.waypoints[self.current_waypoint_idx]

        sphere_goal_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere_goal")
        self.model.geom_pos[sphere_goal_geom_id, :] = self.waypoints[sphere_goal_idx]
        # Reset velocities
        self.data.qvel[:] = 0

        # Update simulation
        mujoco.mj_forward(self.model, self.data)

        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        self.x_positions = []
        self.y_positions = []
        self.theta_positions = []
        self.lin_vel = []
        self.ang_vel = []
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        return observation, info

    def _find_valid_sphere_position(self, cube_x, cube_y, min_x, max_x, min_y, max_y, min_separation):
        """Find a valid position for the sphere that maintains minimum separation from cube."""
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            sphere_x = np.random.uniform(min_x, max_x)
            sphere_y = np.random.uniform(min_y, max_y)

            if np.sqrt((cube_x - sphere_x) ** 2 + (cube_y - sphere_y) ** 2) >= min_separation:
                return sphere_x, sphere_y
            attempts += 1

        # Fallback - place sphere at fixed distance
        angle = np.random.uniform(0, 2 * np.pi)
        sphere_x = cube_x + min_separation * np.cos(angle)
        sphere_y = cube_y + min_separation * np.sin(angle)
        return np.clip(sphere_x, min_x, max_x), np.clip(sphere_y, min_y, max_y)

    # def _empty(self):
    #     wall_buffer = 0.2
    #     min_x = -7.5 + wall_buffer
    #     max_x = 7.5 - wall_buffer
    #     min_y = -7.5 + wall_buffer
    #     max_y = 7.5 - wall_buffer
    #     cube_x = np.random.uniform(min_x + 0.32, max_x - 0.32)
    #     cube_y = np.random.uniform(min_y + 0.32, max_y - 0.32)
    #     sphere_x, sphere_y = self._find_valid_sphere_position(
    #         cube_x, cube_y, min_x + 1, max_x - 1, min_y + 1, max_y - 1, 5
    #     )
    #     return cube_x, cube_y, sphere_x, sphere_y


    # def _obstacles(self):
    #     wall_buffer = 0.2
    #     min_x = -7.5 + wall_buffer
    #     max_x = 7.5 - wall_buffer
    #     min_y = -7.5 + wall_buffer
    #     max_y = 7.5 - wall_buffer
    #
    #     resolution = 0.1
    #     min_separation = 5.0  # Minimum distance between objects
    #     max_placement_attempts = 100
    #     placement_attempts = 0
    #
    #     OBSTACLE_TYPES = {
    #         "small_box": {"type": "box", "size": [0.5, 0.5, 1], "mass": 1.0},
    #         "mid_box_x": {"type": "box", "size": [1, 0.5, 1.0], "mass": 1.0},
    #         "mid_box_y": {"type": "box", "size": [0.5, 1, 1.0], "mass": 1.0},
    #         "big_box_x": {"type": "box", "size": [1.5, 0.5, 1.0], "mass": 1.0},
    #         "big_box_y": {"type": "box", "size": [0.5, 1.5, 1.0], "mass": 1.0},
    #     }
    #
    #     # 1. Place cube randomly
    #     cube_x = np.random.uniform(min_x + 0.32, max_x - 0.32)
    #     cube_y = np.random.uniform(min_y + 0.32, max_y - 0.32)
    #     # cube_yaw = np.random.uniform(-np.pi, np.pi)
    #
    #     # 2. Place sphere with minimum separation from cube
    #     sphere_x, sphere_y = self._find_valid_sphere_position(
    #         cube_x, cube_y, min_x + 1, max_x - 1, min_y + 1, max_y - 1, min_separation
    #     )
    #
    #     path_exists = False
    #     # 3. Place obstacles with checks against both cube and sphere
    #     while not path_exists and placement_attempts < max_placement_attempts:
    #         grid = np.zeros((150, 150), dtype=np.int8)
    #         for i in range(1, 21):
    #             obstacle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}")
    #             if obstacle_body_id >= 0:
    #                 """Pick random size for the obstacle"""
    #                 obs_name = np.random.choice(list(OBSTACLE_TYPES.keys()))
    #                 obs_data = OBSTACLE_TYPES[obs_name]
    #                 obs_half_x = obs_data["size"][0]
    #                 obs_half_y = obs_data["size"][1]
    #
    #                 # Find valid position for this obstacle
    #                 obstacle_x, obstacle_y, obs_size_x, obs_size_y = self._find_valid_obstacle_position(
    #                     i, cube_x, cube_y, sphere_x, sphere_y,
    #                     min_x + obs_half_x, max_x - obs_half_x, min_y + obs_half_y, max_y - obs_half_x, min_separation,
    #                     obs_half_x, obs_half_y
    #                 )
    #                 self.model.body_pos[obstacle_body_id, :2] = [obstacle_x, obstacle_y]
    #                 gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle{i}")
    #                 self.model.geom_size[gid, 0] = obs_size_x
    #                 self.model.geom_size[gid, 1] = obs_size_y
    #
    #
    #                 min_x_world = obstacle_x - obs_size_x - 0.36
    #                 max_x_world = obstacle_x + obs_size_x + 0.36
    #                 min_y_world = obstacle_y - obs_size_y - 0.36
    #                 max_y_world = obstacle_y + obs_size_y + 0.36
    #
    #                 shifted_min_x = min_x_world + 7.5
    #                 shifted_max_x = max_x_world + 7.5
    #                 shifted_min_y = min_y_world + 7.5
    #                 shifted_max_y = max_y_world + 7.5
    #
    #                 # 3. Convert shifted world coordinates to grid indices (clamping to stay within 0-150)
    #                 min_col = np.clip(int(shifted_min_x / resolution), 0, 150)
    #                 max_col = np.clip(int(shifted_max_x / resolution), 0, 150)
    #                 min_row = np.clip(int(shifted_min_y / resolution), 0, 150)
    #                 max_row = np.clip(int(shifted_max_y / resolution), 0, 150)
    #                 # 4. Mark the inflated region as occupied (1)
    #                 grid[min_row:max_row, min_col:max_col] = 1
    #
    #         start_node = world_to_grid((cube_x, cube_y))
    #         goal_node = world_to_grid((sphere_x, sphere_y))
    #         # 3. Run A* Search
    #         path_exists = run_astar(grid, start_node, goal_node)
    #     # Fallback configuration if random placement fails
    #     if not path_exists:
    #         cube_x, cube_y = 0, 0  # Center position
    #         # cube_yaw = np.random.uniform(-np.pi, np.pi)
    #         sphere_x, sphere_y = 5, 0  # East position
    #         self._place_obstacles_in_circle(radius=10)
    #     return cube_x, cube_y, sphere_x, sphere_y


    # def _door_configuration(self):
    #     # Need two obstacles like walls which are positioned in a way that makes a door which is wide 1 meter
    #     # The door can be either in the x direction or in the y direction
    #     # Also the door can be moved up and down and left and right
    #
    #     # Obstacles for y direction. Pick random x and y dor the door center between (-5, 5)
    #     door_x = np.random.uniform(-5.0, 5.0)
    #     door_y = np.random.uniform(-5.0, 5.0)
    #     obs_ids = []
    #     counter = 0
    #     for i in range(1, 10):
    #         obstacle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}")
    #         if obstacle_body_id >= 0:
    #             counter += 1
    #             obs_ids.append(obstacle_body_id)
    #         if counter == 2:
    #             break
    #     if np.random.uniform(0, 1) < 0.5:
    #         # Walls in Y direction
    #         # Wall 1
    #         pos1_x = door_x
    #         dist1_y = (7.5 - door_y) / 2
    #         pos1_y = 7.5 - dist1_y + 0.5
    #
    #         self.model.body_pos[obs_ids[0], :2] = [pos1_x, pos1_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle1")
    #         self.model.geom_size[gid, 0] = 0.25
    #         self.model.geom_size[gid, 1] = 7.5 - pos1_y
    #
    #         # Wall 2
    #         pos2_x = door_x
    #         dist2_y = (-7.5 - door_y) / 2
    #         pos2_y = -7.5 - dist2_y - 0.5
    #         self.model.body_pos[obs_ids[1], :2] = [pos2_x, pos2_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle2")
    #         self.model.geom_size[gid, 0] = 0.25
    #         self.model.geom_size[gid, 1] = abs(-7.5 - pos2_y)
    #
    #         if np.random.uniform(0, 1) < 0.5:
    #             # Goal above
    #             sphere_x = np.random.uniform(door_x + 1.5, 6.5)
    #             sphere_y = np.random.uniform(door_y - 2, door_y + 2)
    #
    #             cube_x = np.random.uniform(-6.5, door_x - 1.5)
    #             cube_y = np.random.uniform(door_y - 2, door_y + 2)
    #         else:
    #             # Goal below
    #             sphere_x = np.random.uniform(-6.5, door_x - 1.5)
    #             sphere_y = np.random.uniform(door_y - 2, door_y + 2)
    #
    #             cube_x = np.random.uniform(door_x + 1.5, 6.5)
    #             cube_y = np.random.uniform(door_y - 2, door_y + 2)
    #         return cube_x, cube_y, sphere_x, sphere_y
    #     else:
    #         # Walls in X direction
    #         # Wall 1
    #         pos1_y = door_y
    #         dist1_x = (7.5 - door_x) / 2
    #         pos1_x = 7.5 - dist1_x + 0.5
    #
    #         self.model.body_pos[obs_ids[0], :2] = [pos1_x, pos1_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle1")
    #         self.model.geom_size[gid, 0] = 7.5 - pos1_x
    #         self.model.geom_size[gid, 1] = 0.25
    #
    #         # Wall 2
    #         pos2_y = door_y
    #         dist2_x = (-7.5 - door_x) / 2
    #         pos2_x = -7.5 - dist2_x - 0.5
    #         self.model.body_pos[obs_ids[1], :2] = [pos2_x, pos2_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle2")
    #         self.model.geom_size[gid, 0] = abs(-7.5 - pos2_x)
    #         self.model.geom_size[gid, 1] = 0.25
    #
    #         if np.random.uniform(0, 1) < 0.5:
    #             # Goal left
    #             sphere_x = np.random.uniform(door_x - 2, door_x + 2)
    #             sphere_y = np.random.uniform(door_y + 1.5, 6.5)
    #
    #             cube_x = np.random.uniform(door_x - 2, door_x + 2)
    #             cube_y = np.random.uniform(-6.5, door_y - 1.5)
    #         else:
    #             # Goal right
    #             sphere_x = np.random.uniform(door_x - 2, door_x + 2)
    #             sphere_y = np.random.uniform(-6.5, door_y - 1.5)
    #
    #             cube_x = np.random.uniform(door_x - 2, door_x + 2)
    #             cube_y = np.random.uniform(door_y + 1.5, 6.5)
    #         return cube_x, cube_y, sphere_x, sphere_y


    # def _single_angle_configuration(self):
    #     # We need two walls which make 90 deg angle
    #     # Randomly pick the location of the corner and set the two walls
    #
    #     corner_x = np.random.uniform(-5.0, 5.0)
    #     corner_y = np.random.uniform(-5.0, 5.0)
    #     obs_ids = []
    #     counter = 0
    #     for i in range(1, 10):
    #         obstacle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}")
    #         if obstacle_body_id >= 0:
    #             counter += 1
    #             obs_ids.append(obstacle_body_id)
    #         if counter == 2:
    #             break
    #     # Case 1 Г
    #     choice = np.random.uniform(0, 1)
    #     if choice < 0.25:
    #         # Case 1 Г
    #         # Wall 1
    #         pos1_x = corner_x
    #         dist1_y = (-7.5 - corner_y) / 2
    #         pos1_y = -7.5 - dist1_y
    #
    #         self.model.body_pos[obs_ids[0], :2] = [pos1_x, pos1_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle1")
    #         self.model.geom_size[gid, 0] = 0.25
    #         self.model.geom_size[gid, 1] = abs(-7.5 - pos1_y)
    #
    #
    #         # Wall 2
    #         pos2_y = corner_y
    #         dist2_x = (-7.5 - corner_x) / 2
    #         pos2_x = -7.5 - dist2_x
    #         self.model.body_pos[obs_ids[1], :2] = [pos2_x, pos2_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle2")
    #         self.model.geom_size[gid, 0] = abs(-7.5 - pos2_x - 0.25)
    #         self.model.geom_size[gid, 1] = 0.25
    #
    #         if np.random.uniform(0, 1) < 0.5:
    #             # Goal up right
    #             sphere_x = np.random.uniform(corner_x + 1, 6.5)
    #             sphere_y = np.random.uniform(- 6.5, corner_y + 2)
    #
    #
    #             cube_x = np.random.uniform(-6.5, corner_x - 1)
    #             cube_y = np.random.uniform(corner_y + 2, 6.5)
    #
    #
    #         else:
    #             # Goal down left
    #             sphere_x = np.random.uniform(-6.5, corner_x - 1)
    #             sphere_y = np.random.uniform(corner_y + 2, 6.5)
    #
    #             cube_x = np.random.uniform(corner_x + 1, 6.5)
    #             cube_y = np.random.uniform(- 6.5, corner_y + 2)
    #         return cube_x, cube_y, sphere_x, sphere_y
    #     elif 0.25 <= choice < 0.5:
    #         # Case 2 _|
    #         # Wall 1
    #         pos1_x = corner_x
    #         dist1_y = (7.5 - corner_y) / 2
    #         pos1_y = 7.5 - dist1_y
    #
    #         self.model.body_pos[obs_ids[0], :2] = [pos1_x, pos1_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle1")
    #         self.model.geom_size[gid, 0] = 0.25
    #         self.model.geom_size[gid, 1] = 7.5 - pos1_y
    #
    #         # Wall 2
    #         pos2_y = corner_y
    #         dist2_x = (7.5 - corner_x) / 2
    #         pos2_x = 7.5 - dist2_x
    #         self.model.body_pos[obs_ids[1], :2] = [pos2_x, pos2_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle2")
    #         self.model.geom_size[gid, 0] = 7.5 - pos2_x + 0.25
    #         self.model.geom_size[gid, 1] = 0.25
    #
    #         if np.random.uniform(0, 1) < 0.5:
    #             # Goal up right
    #             sphere_x = np.random.uniform(corner_x + 1, 6.5)
    #             sphere_y = np.random.uniform(-6.5, corner_y - 2)
    #
    #             cube_x = np.random.uniform(-6.5, corner_x - 1)
    #             cube_y = np.random.uniform(corner_y + 2, 6.5)
    #         else:
    #             # Goal down left
    #             sphere_x = np.random.uniform(-6.5, corner_x - 1)
    #             sphere_y = np.random.uniform(corner_y + 2, 6.5)
    #
    #             cube_x = np.random.uniform(corner_x + 1, 6.5)
    #             cube_y = np.random.uniform(-6.5, corner_y - 2)
    #         return cube_x, cube_y, sphere_x, sphere_y
    #     elif 0.5 <= choice < 0.75:
    #
    #         # Case 3 |_
    #         # Wall 1
    #         pos1_x = corner_x
    #         dist1_y = (-7.5 - corner_y) / 2
    #         pos1_y = -7.5 - dist1_y
    #
    #         self.model.body_pos[obs_ids[0], :2] = [pos1_x, pos1_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle1")
    #         self.model.geom_size[gid, 0] = 0.25
    #         self.model.geom_size[gid, 1] = abs(-7.5 - pos1_y)
    #
    #         # Wall 2
    #         pos2_y = corner_y
    #         dist2_x = (7.5 - corner_x) / 2
    #         pos2_x = 7.5 - dist2_x
    #         self.model.body_pos[obs_ids[1], :2] = [pos2_x, pos2_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle2")
    #         self.model.geom_size[gid, 0] = 7.5 - pos2_x + 0.25
    #         self.model.geom_size[gid, 1] = 0.25
    #
    #         if np.random.uniform(0, 1) < 0.5:
    #             # Goal up left
    #             sphere_x = np.random.uniform(corner_x + 1, 6.5)
    #             sphere_y = np.random.uniform(corner_y + 2, 6.5)
    #
    #             cube_x = np.random.uniform(-6.5, corner_x - 1)
    #             cube_y = np.random.uniform(-6.5, corner_y - 2)
    #         else:
    #             # Goal down right
    #             sphere_x = np.random.uniform(-6.5, corner_x - 1)
    #             sphere_y = np.random.uniform(-6.5, corner_y - 2)
    #
    #             cube_x = np.random.uniform(corner_x + 1, 6.5)
    #             cube_y = np.random.uniform(corner_y + 2, 6.5)
    #         return cube_x, cube_y, sphere_x, sphere_y
    #
    #     else:
    #         # Case 4 -^|
    #         # Wall 1
    #         pos1_x = corner_x
    #         dist1_y = (7.5 - corner_y) / 2
    #         pos1_y = 7.5 - dist1_y
    #
    #         self.model.body_pos[obs_ids[0], :2] = [pos1_x, pos1_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle1")
    #         self.model.geom_size[gid, 0] = 0.25
    #         self.model.geom_size[gid, 1] = 7.5 - pos1_y
    #
    #         # Wall 2
    #         pos2_y = corner_y
    #         dist2_x = (-7.5 - corner_x) / 2
    #         pos2_x = -7.5 - dist2_x
    #         self.model.body_pos[obs_ids[1], :2] = [pos2_x, pos2_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle2")
    #         self.model.geom_size[gid, 0] = abs(-7.5 - pos2_x - 0.25)
    #         self.model.geom_size[gid, 1] = 0.25
    #
    #         if np.random.uniform(0, 1) < 0.5:
    #             # Goal up left
    #             sphere_x = np.random.uniform(corner_x + 1, 6.5)
    #             sphere_y = np.random.uniform(corner_y + 2, 6.5)
    #
    #             cube_x = np.random.uniform(-6.5, corner_x - 1)
    #             cube_y = np.random.uniform(-6.5, corner_y - 2)
    #         else:
    #             # Goal down right
    #             sphere_x = np.random.uniform(-6.5, corner_x - 1 )
    #             sphere_y = np.random.uniform(-6.5, corner_y - 2)
    #
    #             cube_x = np.random.uniform(corner_x + 1, 6.5)
    #             cube_y = np.random.uniform(corner_y + 2, 6.5)
    #         return cube_x, cube_y, sphere_x, sphere_y

    # def _two_angles_configuration(self):
    #     corridor_size = np.random.uniform(1.5, 4)
    #     door_x = np.random.uniform(-3, 3)
    #     door_y = np.random.uniform(-3, 3)
    #
    #     obs_ids = []
    #     counter = 0
    #     for i in range(1, 10):
    #         obstacle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}")
    #         if obstacle_body_id >= 0:
    #             counter += 1
    #             obs_ids.append(obstacle_body_id)
    #         if counter == 4:
    #             break
    #
    #     choice_config = np.random.uniform(0,1)
    #
    #     if choice_config < 0.25:
    #         # Case 1 in y directiom
    #         corner1_x = door_x
    #         corner1_y = door_y + corridor_size / 2
    #         corner2_y = door_y - corridor_size / 2
    #
    #         # Corner 1
    #         # Wall 1
    #         pos1_x = corner1_x
    #         dist1_y = (7.5 - corner1_y) / 2
    #         pos1_y = 7.5 - dist1_y
    #
    #         self.model.body_pos[obs_ids[0], :2] = [pos1_x, pos1_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle1")
    #         self.model.geom_size[gid, 0] = 0.25
    #         self.model.geom_size[gid, 1] = 7.5 - pos1_y
    #
    #         # Wall 2
    #         pos2_y = corner1_y
    #         dist2_x = (-7.5 - corner1_x) / 2
    #         pos2_x = -7.5 - dist2_x
    #         self.model.body_pos[obs_ids[1], :2] = [pos2_x, pos2_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle2")
    #         self.model.geom_size[gid, 0] = abs(-7.5 - pos2_x - 0.25)
    #         self.model.geom_size[gid, 1] = 0.25
    #
    #         # Corner 2
    #         # Wall 1
    #         pos1_x = corner1_x
    #         dist1_y = (-7.5 - corner2_y) / 2
    #         pos1_y = -7.5 - dist1_y
    #
    #         self.model.body_pos[obs_ids[2], :2] = [pos1_x, pos1_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle3")
    #         self.model.geom_size[gid, 0] = 0.25
    #         self.model.geom_size[gid, 1] = abs(-7.5 - pos1_y)
    #
    #         # Wall 2
    #         pos2_y = corner2_y
    #         dist2_x = (-7.5 - corner1_x) / 2
    #         pos2_x = -7.5 - dist2_x
    #         self.model.body_pos[obs_ids[3], :2] = [pos2_x, pos2_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle4")
    #         self.model.geom_size[gid, 0] = abs(-7.5 - pos2_x - 0.25)
    #         self.model.geom_size[gid, 1] = 0.25
    #         choice = np.random.uniform(0,1)
    #         if choice < 0.25:
    #             # Goal up right
    #             sphere_x = np.random.uniform(corner1_x + 1, 6.5)
    #             sphere_y = np.random.uniform(-6.5, corner2_y - 2)
    #
    #             cube_x = np.random.uniform(-6.5, door_x - 1)
    #             cube_y = door_y
    #         elif 0.25 < choice < 0.5:
    #             # Goal up left
    #             sphere_x = np.random.uniform(corner1_x + 1, 6.5)
    #             sphere_y = np.random.uniform(corner1_y + 2, 6.5)
    #
    #             cube_x = np.random.uniform(-6.5, door_x - 1)
    #             cube_y = door_y
    #         elif 0.5 < choice < 0.75:
    #             # Goal in corridor start right
    #             cube_x = np.random.uniform(corner1_x + 1, 6.5)
    #             cube_y = np.random.uniform(- 6.5, corner2_y - 2)
    #
    #             sphere_x = np.random.uniform(-6.5, door_x - 1)
    #             sphere_y = door_y
    #         else:
    #             # Goal in corridor start left
    #             cube_x = np.random.uniform(corner1_x + 1, 6.5)
    #             cube_y = np.random.uniform(corner1_y + 2, 6.5)
    #
    #             sphere_x = np.random.uniform(-6.5, door_x - 1)
    #             sphere_y = door_y
    #
    #         return cube_x, cube_y, sphere_x, sphere_y
    #     elif 0.25 < choice_config < 0.5:
    #         # Case 2 in y directiom
    #         corner1_x = door_x
    #         corner1_y = door_y + corridor_size / 2
    #         corner2_y = door_y - corridor_size / 2
    #
    #         # Corner 1
    #         # Wall 1
    #         pos1_x = corner1_x
    #         dist1_y = (7.5 - corner1_y) / 2
    #         pos1_y = 7.5 - dist1_y
    #
    #         self.model.body_pos[obs_ids[0], :2] = [pos1_x, pos1_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle1")
    #         self.model.geom_size[gid, 0] = 0.25
    #         self.model.geom_size[gid, 1] = 7.5 - pos1_y
    #
    #         # Wall 2
    #         pos2_y = corner1_y
    #         dist2_x = (7.5 - corner1_x) / 2
    #         pos2_x = 7.5 - dist2_x
    #         self.model.body_pos[obs_ids[1], :2] = [pos2_x, pos2_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle2")
    #         self.model.geom_size[gid, 0] = 7.5 - pos2_x + 0.25
    #         self.model.geom_size[gid, 1] = 0.25
    #
    #         # Corner 2
    #         # Wall 1
    #         pos1_x = corner1_x
    #         dist1_y = (-7.5 - corner2_y) / 2
    #         pos1_y = -7.5 - dist1_y
    #
    #         self.model.body_pos[obs_ids[2], :2] = [pos1_x, pos1_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle3")
    #         self.model.geom_size[gid, 0] = 0.25
    #         self.model.geom_size[gid, 1] = abs(-7.5 - pos1_y)
    #
    #         # Wall 2
    #         pos2_y = corner2_y
    #         dist2_x = (7.5 - corner1_x) / 2
    #         pos2_x = 7.5 - dist2_x
    #         self.model.body_pos[obs_ids[3], :2] = [pos2_x, pos2_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle4")
    #         self.model.geom_size[gid, 0] = 7.5 - pos2_x + 0.25
    #         self.model.geom_size[gid, 1] = 0.25
    #         choice = np.random.uniform(0, 1)
    #         if choice < 0.25:
    #             # Goal down right
    #             sphere_x = np.random.uniform(-6.5, corner1_x - 1)
    #             sphere_y = np.random.uniform(-6.5, corner2_y - 2)
    #
    #             cube_x = np.random.uniform(door_x + 1, 6.5)
    #             cube_y = door_y
    #         elif 0.25 < choice < 0.5:
    #             # Goal down left
    #             sphere_x = np.random.uniform(-6.5, corner1_x - 1)
    #             sphere_y = np.random.uniform(corner1_y + 2, 6.5)
    #
    #             cube_x = np.random.uniform(door_x + 1, 6.5)
    #             cube_y = door_y
    #         elif 0.5 < choice < 0.75:
    #             # Goal in corridor start right
    #             cube_x = np.random.uniform(-6.5, corner1_x - 1)
    #             cube_y = np.random.uniform( -6.5, corner2_y - 2)
    #
    #             sphere_x = np.random.uniform(door_x + 1, 6.5)
    #             sphere_y = door_y
    #         else:
    #             # Goal in corridor start left
    #             cube_x = np.random.uniform(-6.5, corner1_x - 1)
    #             cube_y = np.random.uniform(corner1_y + 2, 6.5)
    #
    #             sphere_x = np.random.uniform(door_x + 1, 6.5)
    #             sphere_y = door_y
    #         return cube_x, cube_y, sphere_x, sphere_y
    #     elif 0.5 < choice_config < 0.75:
    #         # Case 1 in x directiom
    #         corner1_x = door_x + corridor_size / 2
    #         corner2_x = door_x - corridor_size / 2
    #         corner1_y = door_y
    #
    #         # Corner 1
    #         # Wall 1
    #         pos1_x = corner1_x
    #         dist1_y = (7.5 - corner1_y) / 2
    #         pos1_y = 7.5 - dist1_y
    #
    #         self.model.body_pos[obs_ids[0], :2] = [pos1_x, pos1_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle1")
    #         self.model.geom_size[gid, 0] = 0.25
    #         self.model.geom_size[gid, 1] = 7.5 - pos1_y
    #
    #         # Wall 2
    #         pos2_y = corner1_y
    #         dist2_x = (7.5 - corner1_x) / 2
    #         pos2_x = 7.5 - dist2_x
    #         self.model.body_pos[obs_ids[1], :2] = [pos2_x, pos2_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle2")
    #         self.model.geom_size[gid, 0] = 7.5 - pos2_x + 0.25
    #         self.model.geom_size[gid, 1] = 0.25
    #
    #         # Corner 2
    #         # Wall 1
    #         pos1_x = corner2_x
    #         dist1_y = (7.5 - corner1_y) / 2
    #         pos1_y = 7.5 - dist1_y
    #
    #         self.model.body_pos[obs_ids[2], :2] = [pos1_x, pos1_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle3")
    #         self.model.geom_size[gid, 0] = 0.25
    #         self.model.geom_size[gid, 1] = 7.5 - pos1_y
    #
    #         # Wall 2
    #         pos2_y = corner1_y
    #         dist2_x = (-7.5 - corner2_x) / 2
    #         pos2_x = -7.5 - dist2_x
    #
    #         self.model.body_pos[obs_ids[3], :2] = [pos2_x, pos2_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle4")
    #         self.model.geom_size[gid, 0] = abs(-7.5 - pos2_x - 0.25)
    #         self.model.geom_size[gid, 1] = 0.25
    #         choice = np.random.uniform(0, 1)
    #         if choice < 0.25:
    #             # Goal down right
    #             sphere_x = np.random.uniform(-6.5, corner2_x - 1)
    #             sphere_y = np.random.uniform(-6.5, corner1_y - 2)
    #
    #             cube_x = door_x
    #             cube_y = np.random.uniform(door_y + 1, 6.5)
    #         elif 0.25 < choice < 0.5:
    #             # Goal up right
    #             sphere_x = np.random.uniform(corner1_x + 1, 6.5)
    #             sphere_y = np.random.uniform( -6.5, corner1_y - 2)
    #
    #             cube_x = door_x
    #             cube_y = np.random.uniform(door_y + 1, 6.5)
    #         elif 0.5 < choice < 0.75:
    #             # Goal in corridor start down right
    #             cube_x = np.random.uniform(-6.5, corner1_x - 1)
    #             cube_y = np.random.uniform(-6.5, corner1_y - 2)
    #
    #             sphere_x = door_x
    #             sphere_y = np.random.uniform(door_y + 1, 6.5)
    #         else:
    #             # Goal in corridor start up right
    #             cube_x = np.random.uniform(corner1_x + 1, 6.5)
    #             cube_y = np.random.uniform(-6.5, corner1_y - 2)
    #
    #             sphere_x = door_x
    #             sphere_y = np.random.uniform(door_y + 1, 6.5)
    #         return cube_x, cube_y, sphere_x, sphere_y
    #     else:
    #         # Case 1 in x directiom
    #         corner1_x = door_x + corridor_size / 2
    #         corner2_x = door_x - corridor_size / 2
    #         corner1_y = door_y
    #
    #         # Corner 1
    #         # Wall 1
    #         pos1_x = corner1_x
    #         dist1_y = (-7.5 - corner1_y) / 2
    #         pos1_y = -7.5 - dist1_y
    #
    #         self.model.body_pos[obs_ids[0], :2] = [pos1_x, pos1_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle1")
    #         self.model.geom_size[gid, 0] = 0.25
    #         self.model.geom_size[gid, 1] = abs(-7.5 - pos1_y)
    #
    #         # Wall 2
    #         pos2_y = corner1_y
    #         dist2_x = (7.5 - corner1_x) / 2
    #         pos2_x = 7.5 - dist2_x
    #         self.model.body_pos[obs_ids[1], :2] = [pos2_x, pos2_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle2")
    #         self.model.geom_size[gid, 0] = 7.5 - pos2_x + 0.25
    #         self.model.geom_size[gid, 1] = 0.25
    #
    #         # Corner 2
    #         # Wall 1
    #         pos1_x = corner2_x
    #         dist1_y = (-7.5 - corner1_y) / 2
    #         pos1_y = -7.5 - dist1_y
    #
    #         self.model.body_pos[obs_ids[2], :2] = [pos1_x, pos1_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle3")
    #         self.model.geom_size[gid, 0] = 0.25
    #         self.model.geom_size[gid, 1] = abs(-7.5 - pos1_y)
    #
    #         # Wall 2
    #         pos2_y = corner1_y
    #         dist2_x = (-7.5 - corner2_x) / 2
    #         pos2_x = -7.5 - dist2_x
    #
    #         self.model.body_pos[obs_ids[3], :2] = [pos2_x, pos2_y]
    #         gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle4")
    #         self.model.geom_size[gid, 0] = abs(-7.5 - pos2_x - 0.25)
    #         self.model.geom_size[gid, 1] = 0.25
    #         choice = np.random.uniform(0, 1)
    #         if choice < 0.25:
    #             # Goal down left
    #             sphere_x = np.random.uniform(-6.5, corner2_x - 1)
    #             sphere_y = np.random.uniform(corner1_y + 2, 6.5)
    #
    #             cube_x = door_x
    #             cube_y = np.random.uniform(-6.5, door_y - 1)
    #         elif 0.25 < choice < 0.5:
    #             # Goal up left
    #             sphere_x = np.random.uniform(corner1_x + 1, 6.5)
    #             sphere_y = np.random.uniform(corner1_y + 2, 6.5)
    #
    #             cube_x = door_x
    #             cube_y = np.random.uniform(-6.5, door_y - 1)
    #         elif 0.5 < choice < 0.75:
    #             # Goal in corridor start down left
    #             cube_x = np.random.uniform(-6.5, corner2_x - 1)
    #             cube_y = np.random.uniform(corner1_y + 2, 6.5)
    #
    #             sphere_x = door_x
    #             sphere_y = np.random.uniform(-6.5, door_y - 1)
    #         else:
    #             # Goal in corridor start up left
    #             cube_x = np.random.uniform(corner1_x + 1, 6.5)
    #             cube_y = np.random.uniform(corner1_y + 2, 6.5)
    #
    #             sphere_x = door_x
    #             sphere_y = np.random.uniform(-6.5, door_y - 1)
    #         return cube_x, cube_y, sphere_x, sphere_y


    # def _find_valid_obstacle_position(self, obstacle_num, cube_x, cube_y, sphere_x, sphere_y,
    #                                   min_x, max_x, min_y, max_y, min_separation, obs_half_x, obs_half_y):
    #     """Find valid position for a specific obstacle."""
    #     attempts = 0
    #     max_attempts = 100
    #
    #     while attempts < max_attempts:
    #         # Pick x and y coordinates
    #         x = np.random.uniform(min_x, max_x)
    #         y = np.random.uniform(min_y, max_y)
    #
    #
    #         # 2. Find the closest point (Px, Py) on the obstacle to the robot's center
    #         cube_closest_x = np.clip(cube_x, x - obs_half_x, x + obs_half_x)
    #         cube_closest_y = np.clip(cube_y, y - obs_half_y, y + obs_half_y)
    #
    #         # 3. Calculate the distance squared between the robot center and the closest point
    #         # Note: We can avoid the expensive sqrt by comparing distance squared (D^2 < R^2)
    #         cube_dist_x_sq = (cube_closest_x - cube_x) ** 2
    #         cube_dist_y_sq = (cube_closest_y - cube_y) ** 2
    #         cube_distance_squared = cube_dist_x_sq + cube_dist_y_sq
    #         # the radius of the robot is 0.32 + safety margin(0.3)
    #         cube_required_radius = (0.32 + 0.3)**2
    #         if cube_distance_squared < cube_required_radius:
    #             attempts += 1
    #             continue
    #
    #         # 2. Find the closest point (Px, Py) on the obstacle to the sphere's center
    #         sphere_closest_x = np.clip(sphere_x, x - obs_half_x, x + obs_half_x)
    #         sphere_closest_y = np.clip(sphere_y, y - obs_half_y, y + obs_half_y)
    #
    #         # 3. Calculate the distance squared between the sphere center and the closest point
    #         # Note: We can avoid the expensive sqrt by comparing distance squared (D^2 < R^2)
    #         sphere_dist_x_sq = (sphere_closest_x - sphere_x) ** 2
    #         sphere_dist_y_sq = (sphere_closest_y - sphere_y) ** 2
    #         sphere_distance_squared = sphere_dist_x_sq + sphere_dist_y_sq
    #         # the radius of the sphere is 1 + safety margin(0.3)
    #         sphere_required_radius = (1 + 0.2) ** 2
    #         if sphere_distance_squared < sphere_required_radius:
    #             attempts += 1
    #             continue
    #
    #         # Check distance to other obstacles
    #         too_close = False
    #         for i in range(1, obstacle_num):
    #             other_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}")
    #             if other_id >= 0:
    #                 other_pos = self.model.body_pos[other_id].copy()
    #                 gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle{i}")
    #                 geom_size_data = self.model.geom_size[gid]
    #                 half_size_x = geom_size_data[0]
    #                 half_size_y = geom_size_data[1]
    #
    #                 check_x = abs(x - other_pos[0]) < obs_half_x + half_size_x
    #                 check_y = abs(y - other_pos[1]) < obs_half_y + half_size_y
    #
    #                 if check_x and check_y:
    #                     too_close = True
    #                     break
    #
    #         if not too_close:
    #             return x, y, obs_half_x, obs_half_y
    #
    #         attempts += 1
    #     # Fallback position
    #     angle = (obstacle_num-1) * (2*np.pi/20)
    #     radius = 10.0
    #     x = radius * np.cos(angle)
    #     y = radius * np.sin(angle)
    #     return np.clip(x, min_x, max_x), np.clip(y, min_y, max_y), 1, 1


    # def _place_obstacles_in_circle(self, radius=10.0):
    #     """Place all obstacles in a circle pattern as fallback."""
    #     for i in range(1, 21):
    #         obstacle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}")
    #         if obstacle_body_id >= 0:
    #             angle = (i - 1) * (2 * np.pi / 20)
    #             x = radius * np.cos(angle)
    #             y = radius * np.sin(angle)
    #             self.model.body_pos[obstacle_body_id, :2] = [x, y]

    def step(self, action):
        # Movement parameters
        move_max_linspeed = 0.25
        max_ang_speed = 1
        # Apply action
        target_linear_velocity = float(action[0]) * move_max_linspeed
        target_angular_velocity = float(action[1]) * max_ang_speed

        alpha_v = 0.15  # linear response
        alpha_w = 0.15  # angular response
        # store previous velocities for smoothness reward
        prev_lin_vel = self.linear_velocity
        prev_ang_vel = self.angular_velocity

        self.linear_velocity += alpha_v * (
                target_linear_velocity - self.linear_velocity
        )

        self.angular_velocity += alpha_w * (
                target_angular_velocity - self.angular_velocity
        )

        allowed_ang_vel = max_ang_speed * (1.0 - abs(self.linear_velocity) / move_max_linspeed)

        allowed_ang_vel = max(0.0, allowed_ang_vel)

        self.angular_velocity = np.clip(self.angular_velocity, -allowed_ang_vel, allowed_ang_vel)

        allowed_lin_vel = move_max_linspeed * (1.0 - abs(self.angular_velocity) / max_ang_speed)

        allowed_lin_vel = max(0.0, allowed_lin_vel)

        self.linear_velocity = np.clip(self.linear_velocity,0.0,allowed_lin_vel)

        linear_velocity = self.linear_velocity
        angular_velocity = self.angular_velocity

        # # Triangular constrain
        # allowed_ang_vel = max_ang_speed * (1 - linear_velocity / move_max_linspeed)
        # angular_velocity = np.clip(angular_velocity, -allowed_ang_vel, allowed_ang_vel)
        #
        # allowed_lin_vel = move_max_linspeed * (1 - abs(angular_velocity) / max_ang_speed)
        # linear_velocity = np.clip(linear_velocity, 0.0, allowed_lin_vel)


        # Get current position and orientation
        x, y, theta = self.data.qpos[:3]
        dt = self.model.opt.timestep  # 0.1 [s/step]
        if (np.abs(angular_velocity) > 1e-3):
            deltax = (linear_velocity / angular_velocity) * (np.sin(theta + angular_velocity * dt) - np.sin(theta))
            deltay = (linear_velocity / angular_velocity) * (-np.cos(theta + angular_velocity * dt) + np.cos(theta))
            x += deltax
            y += deltay
            theta += angular_velocity * dt
        else:
            # Updating position
            x += linear_velocity * np.cos(theta) * dt
            y += linear_velocity * np.sin(theta) * dt
        self.x_positions.append(x)
        self.y_positions.append(y)
        self.theta_positions.append(theta)
        self.lin_vel.append(linear_velocity)
        self.ang_vel.append(angular_velocity)

        # Set position and orientation
        self.data.qpos[:3] = [x, y, theta]

        # Step simulation
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        # Update simulation state
        mujoco.mj_forward(self.model, self.data)

        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        reward = 0.0
        # Check for collision with obstacles
        contact_with_obstacles = False
        contact_with_sphere = False



        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            if geom1_id == cube_geom_id or geom2_id == cube_geom_id:
                other_geom_id = geom2_id if geom1_id == cube_geom_id else geom1_id
                other_geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom_id)

                if (other_geom_name and "wall" in other_geom_name) or (
                        other_geom_name and "obstacle" in other_geom_name):
                    contact_with_obstacles = True
                    break
                elif other_geom_name and "sphere" in other_geom_name:
                    contact_with_sphere = True
                    break

        # Calculate reward and check termination conditions
        distance_to_sphere = info["distance_to_sphere"]

        MAX_DIST_STEP = 0.5  # tune (typical robot movement per step)

        reward += 2.0 * np.clip((self.previous_distance - distance_to_sphere) / MAX_DIST_STEP, -1, 1)
        # Find the minimum lidar reading (assuming lidar_readings is a numpy array)

        reward += -0.3 * (abs(self.relative_azimuth) / np.pi)
        # Negative reward as time passes, we want to reach the goal faster without stucking somewhere.
        # I hope that this will make the robot move if he is stuck
        reward -= 0.05 # Time penalty

        reward -= 0.03 * (action[0] ** 2 + action[1] ** 2) # Action penalty

        lin_change = abs(linear_velocity - prev_lin_vel)
        ang_change = abs(angular_velocity - prev_ang_vel)
        reward -= 0.40 * lin_change
        reward -= 0.15 * ang_change
        reward -= 0.02 * abs(angular_velocity)

        # Terminal conditions
        terminated = False
        truncated = False


        too_close_to_obstacles = False
        for i in range(1, len(self.lidar_readings)):
            if self.lidar_readings[i] < 0.32 and self.lidar_readings[i] > 0.07:
                # reward += -0.1 / self.lidar_readings[i]
                d = self.lidar_readings[i]
                reward += -0.3 * np.exp(-d / 0.2)
            elif self.lidar_readings[i] <= 0.15:
                too_close_to_obstacles = True
                break
        self.episode_return += reward

        # if distance_to_sphere < 0.5:
        #     print(f"distance to goal: {distance_to_sphere}")

        if contact_with_obstacles or too_close_to_obstacles:
            reward += -8.0
            self.episode_return += reward
            self.last_episode_result = "collision"
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = time.time() - self.episode_time_begin

            info["x_positions"] = self.x_positions
            info["y_positions"] = self.y_positions
            info["theta_positions"] = self.theta_positions
            info['lin_vel'] = self.lin_vel
            info['ang_vel'] = self.ang_vel
            terminated = True

        # elif distance_to_sphere < 0.3 or contact_with_sphere:
        elif distance_to_sphere < 2.0 or contact_with_sphere:
            self.current_waypoint_idx += 1
            # Check if final waypoint reached

            if self.current_waypoint_idx >= len(self.waypoints):
                reward += 15
                self.episode_return += reward

                self.last_episode_result = "success"
                info["steps_taken"] = self.current_step
                info["episode_time_length"] = time.time() - self.episode_time_begin

                info["x_positions"] = self.x_positions
                info["y_positions"] = self.y_positions
                info["theta_positions"] = self.theta_positions
                info['lin_vel'] = self.lin_vel
                info['ang_vel'] = self.ang_vel
                terminated = True
            else:
                # Move sphere to next waypoint
                next_wp = self.waypoints[self.current_waypoint_idx]
                sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
                self.model.geom_pos[sphere_geom_id, :] = next_wp
                # reset time/steps
                self.current_step = 0
                # Bonus reward for intermediate waypoint
                reward += 15

        elif self.current_step >= self.max_episode_steps:
            self.last_episode_result = "timeout"
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = time.time() - self.episode_time_begin

            info["x_positions"] = self.x_positions
            info["y_positions"] = self.y_positions
            info["theta_positions"] = self.theta_positions
            info['lin_vel'] = self.lin_vel
            info['ang_vel'] = self.ang_vel
            truncated = True

        self.previous_distance = distance_to_sphere

        info["episode_result"] = self.last_episode_result


        # Render if needed
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()
            if not self.training_mode:
                time.sleep(0.01)
                return True
            return True
        return False

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


