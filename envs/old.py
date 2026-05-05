import os
import time
import xml.etree.ElementTree as ET
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

print_one = 0


class CubeNavigationEnv_class(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, num_rays=0, render_mode=None, model_path="xml_models/obstacles_cube_model.xml",
                 training_mode=True):
        super().__init__()

        self.num_rays = num_rays
        self.training_mode = training_mode
        self.max_episode_steps = 1000  # 1000  # 30 seconds (at 30[s]/0.1[s/step] = 300step)
        self.current_step = 0
        self.render_mode = render_mode
        self.model_path = model_path
        self.previous_distance = 30
        self.episode_return = 0
        self.mean_episode_return = 0
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.last_episode_result = None

        self.episode_time_length = 0
        self.episode_time_begin = 0
        self.stuck_counter = 0

        self.lidar_readings = None

        self.sum_progress_reward = 0
        self.average_progress_reward = 0
        self.success_rate = 0
        self.collision_rate = 0
        self.timeout_rate = 0
        self.steps_taken = 0

        self.relative_azimuth = 0

        self.save_lin_vels = []
        self.save_ang_vels = []
        self.max_inference_steps = 0

        # Capire come funziona lo strato di output
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32)

        # Define observation space:
        # - 50 lidar readings (one for each ray)
        # - 1 value for distance to cube (x, y, z)
        # - 1 value for orientation (not used in simple version, but included for future extensions)

        # Kristian's explanation
        # We have say num_rays=50 and the low bound of these measurements is 0, the high is 30.
        # then we have the distance to the cube whose low bound is 0 and high bound is again 30
        # finally we have the orintation which is from -180 to 180 degrees
        # so in total we have num_rays + 2 observations
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.pi]),  #
            high=np.array([30.0, np.pi]),
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

        sensor = None
        for s in root.findall(".//sensor"):
            sensor = s
            break

        if sensor is None:
            raise ValueError("Sensor element not found in the XML model.")

        # Create 360° LiDAR sensors with first ray pointing forward
        for i in range(self.num_rays):
            # Evenly distribute rays around 360°, starting from forward (0°)
            angle = 2 * np.pi * i / self.num_rays

            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            site = ET.SubElement(cube_body, "site")
            site.set("name", f"lidar_site_{i}")
            site.set("pos", f"{-0.1} {0} {0}")
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
        self.viewer.cam.distance = 65.0
        self.viewer.cam.azimuth = 0.0
        self.viewer.cam.elevation = -60.0
        self.viewer.cam.lookat[:] = [0, 0, 1]

    def _get_obs(self):
        # Process LiDAR readings
        self.lidar_readings = np.array([self.data.sensordata[lidar_id] for lidar_id in self.lidar_sensor_ids])

        global print_one
        if print_one == 0:
            print(f'lidar reading shape:{self.lidar_readings.shape}')
            print(self.lidar_readings)
            print_one += 1
        # self.lidar_readings = np.where(self.lidar_readings == -1, 30, self.lidar_readings)
        # print(f"Lidar readings len: {len(self.lidar_readings)}")
        # print(f"Lidar readings: {self.lidar_readings}")
        # print('-----------------------------------------')
        """
        Put noise in the lidar readings based on the distance + modify 5% of the readings to be full scale
        """
        # noise = 0.02
        # print('-----------1-------------')
        # print(self.lidar_readings)
        # # Detect invalid values before generating noise
        # if np.any(self.lidar_readings < 0):
        #     print("NEGATIVE LIDAR VALUE DETECTED:", self.lidar_readings)
        #     print(self.data.sensordata)
        #     raise ValueError("Negative lidar reading before applying noise.")

        # self.lidar_readings = self.lidar_readings + np.random.normal(0, noise*self.lidar_readings)
        # self.lidar_readings = np.clip(self.lidar_readings, 0, 30)
        # print('-----------2-------------')
        # print(self.lidar_readings)

        # Get the reading's indices which are not full scale
        # not_full_idx = np.where(self.lidar_readings < 30)[0]
        # indices = np.random.choice(len(not_full_idx), size=5, replace=False) # to replace
        # self.lidar_readings[not_full_idx[indices]] = 30

        # Get positions
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_body")
        cube_pos = self.data.xpos[cube_body_id].copy()

        sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        sphere_pos = self.data.geom_xpos[sphere_geom_id].copy()

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
        obs = np.array([distance, self.relative_azimuth])
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
        distance_to_sphere = np.linalg.norm(sphere_pos - cube_pos)

        return {
            "distance_to_sphere": distance_to_sphere,
            "cube_position": cube_pos,
            "sphere_position": sphere_pos
        }

    def reset(self, seed=None, options=None):
        self.episode_time_length = time.time() - self.episode_time_begin
        if self.current_step > 0:  # Only set this for non-first episodes
            self.last_episode_info = {"episode_time_length": self.episode_time_length}

        # Update episode statistics
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
        self.previous_distance = 30  # Reset distance tracking
        self.stuck_counter = 0  # Reset stuck counter

        # Define arena boundaries with buffer
        # wall_buffer = 2.0
        min_x = -7.5
        max_x = 7.5
        min_y = -7.5
        max_y = 7.5

        min_separation = 1.0  # Minimum distance between objects
        max_placement_attempts = 100
        placement_attempts = 0
        valid_configuration = False
        OBSTACLE_TYPES = {
            "small_box": {"type": "box", "size": [0.5, 0.5, 1], "mass": 1.0},
            "mid_box_x": {"type": "box", "size": [1, 0.5, 1.0], "mass": 1.0},
            "mid_box_y": {"type": "box", "size": [0.5, 1, 1.0], "mass": 1.0},
            "big_box_x": {"type": "box", "size": [1.5, 0.5, 1.0], "mass": 1.0},
            "big_box_y": {"type": "box", "size": [0.5, 1.5, 1.0], "mass": 1.0},
        }

        while not valid_configuration and placement_attempts < max_placement_attempts:
            # 1. Place cube randomly
            # cube_x = np.random.uniform(min_x, max_x)
            # cube_y = np.random.uniform(min_y, max_y)
            # cube_yaw = np.random.uniform(-np.pi, np.pi)

            cube_x = 0
            cube_y = 0
            cube_yaw = 0

            # 2. Place sphere with minimum separation from cube
            # sphere_x, sphere_y = self._find_valid_sphere_position(
            #     cube_x, cube_y, min_x, max_x, min_y, max_y, min_separation
            # )
            sphere_x, sphere_y = 20, 20
            valid_configuration = True

            # 3. Place obstacles with checks against both cube and sphere
            valid_obstacles = True
            for i in range(1, 21):
                obs_name = np.random.choice(list(OBSTACLE_TYPES.keys()))
                obs_data = OBSTACLE_TYPES[obs_name]
                obs_half_x = obs_data["size"][0]
                obs_half_y = obs_data["size"][1]
                print(f'size:{obs_half_x}, {obs_half_y}, {obs_data["size"]}')
                obstacle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}")
                print(f'i = {i}obstacle id: {obstacle_body_id}')
                if obstacle_body_id >= 0:
                    # Find valid position for this obstacle

                    obstacle_x, obstacle_y = self._find_valid_obstacle_position(
                        i, cube_x, cube_y, sphere_x, sphere_y,
                        min_x + obs_half_x, max_x - obs_half_x, min_y + obs_half_y, max_y - obs_half_y, min_separation,
                        obs_half_x, obs_half_y
                    )

                    # Check if this position is valid relative to cube
                    distance_to_cube = np.sqrt((cube_x - obstacle_x) ** 2 + (cube_y - obstacle_y) ** 2)
                    if distance_to_cube < min_separation:
                        valid_obstacles = False
                        break

                    self.model.body_pos[obstacle_body_id, :2] = [obstacle_x, obstacle_y]
                    obstacle_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle{i}")
                    self.model.geom_size[obstacle_geom_id, :] = obs_data["size"]

            # # Final check that cube isn't too close to any obstacle
            # if valid_obstacles:
            #     valid_configuration = True
            #     for i in range(1, 21):
            #         obstacle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}")
            #         if obstacle_body_id >= 0:
            #             obstacle_pos = self.model.body_pos[obstacle_body_id].copy()
            #             distance = np.sqrt((cube_x - obstacle_pos[0])**2 + (cube_y - obstacle_pos[1])**2)
            #             if distance < min_separation:
            #                 valid_configuration = False
            #                 break

            placement_attempts += 1

        # Fallback configuration if random placement fails
        # if not valid_configuration:
        #     cube_x, cube_y = 0, 0  # Center position
        #     cube_yaw = np.random.uniform(-np.pi, np.pi)
        #     sphere_x, sphere_y = 10, 0  # East position
        #     self._place_obstacles_in_circle(radius=10)  # Place obstacles in a circle

        # Set cube and sphere positions
        self.data.qpos[:3] = [cube_x, cube_y, cube_yaw]
        sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        if sphere_geom_id >= 0:
            self.model.geom_pos[sphere_geom_id, :] = [sphere_x, sphere_y, 2.0]

        # Reset velocities
        self.data.qvel[:] = 0

        # Update simulation
        mujoco.mj_forward(self.model, self.data)

        # Get observation and info
        observation = self._get_obs()
        # print(f'IN reset, obs: {observation}')
        info = self._get_info()

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

    def _find_valid_obstacle_position(self, obstacle_num, cube_x, cube_y, sphere_x, sphere_y,
                                      min_x, max_x, min_y, max_y, min_separation, obs_half_x, obs_half_y):
        """Find valid position for a specific obstacle."""
        attempts = 0
        max_attempts = 100
        ROBOT_SEPARATION = 1.0

        # Separation required for the new obstacle to clear the robot/sphere center
        min_separation_robot = ROBOT_SEPARATION + max(obs_half_x, obs_half_y)

        while attempts < max_attempts:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)

            # Check distance to cube
            if np.sqrt((cube_x - x) ** 2 + (cube_y - y) ** 2) < min_separation_robot:
                attempts += 1
                continue

            # Check distance to sphere
            if np.sqrt((sphere_x - x) ** 2 + (sphere_y - y) ** 2) < min_separation_robot:
                attempts += 1
                continue

            # Check distance to other obstacles
            too_close = False
            for i in range(1, obstacle_num):
                other_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}")
                if other_id >= 0:
                    other_pos = self.model.body_pos[other_id].copy()
                    OTHER_MAX_HALF_EXTENT = 0.75  # Placeholder for the largest half-size of previous obstacles

                    # Required separation: (This obstacle's size) + (Other obstacle's size) + (Safety margin)
                    required_separation_obs = max(obs_half_x, obs_half_y) + OTHER_MAX_HALF_EXTENT
                    if np.sqrt((x - other_pos[0]) ** 2 + (y - other_pos[1]) ** 2) < required_separation_obs:
                        too_close = True
                        break

            if not too_close:
                return x, y

            attempts += 1

        # Fallback position
        angle = (obstacle_num - 1) * (2 * np.pi / 20)
        radius = 10.0
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return np.clip(x, min_x, max_x), np.clip(y, min_y, max_y)

    def _place_obstacles_in_circle(self, radius=10.0):
        """Place all obstacles in a circle pattern as fallback."""
        for i in range(1, 21):
            obstacle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}")
            if obstacle_body_id >= 0:
                angle = (i - 1) * (2 * np.pi / 20)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                self.model.body_pos[obstacle_body_id, :2] = [x, y]

    def step(self, action):

        time.sleep(0.1)
        self.episode_time_length
        # Movement parameters
        move_max_linspeed = 0.25
        max_ang_speed = 1

        # Apply action
        linear_velocity = action[0] * move_max_linspeed
        angular_velocity = action[1] * max_ang_speed

        # Triangular constrain
        allowed_ang_vel = max_ang_speed * (1 - linear_velocity / move_max_linspeed)
        angular_velocity = np.clip(angular_velocity, -allowed_ang_vel, allowed_ang_vel)

        allowed_lin_vel = move_max_linspeed * (1 - abs(angular_velocity) / max_ang_speed)
        linear_velocity = np.clip(linear_velocity, 0.0, allowed_lin_vel)

        # print(f"lin: {allowed_lin_vel}, ang: {angular_velocity}")

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
            # Updating orientation is not necessary, since we are not rotating

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
        # print(f"data.ncon: {self.data.ncon}")
        for i in range(self.data.ncon):
            print('In step(), ncon:')
            contact = self.data.contact[i]
            print(f"contact: {contact}")
            print('---------------')
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            if geom1_id == cube_geom_id or geom2_id == cube_geom_id:
                # Check if the contact is with the walls
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
        # print('Lidar readings:')
        # print(self.lidar_readings)

        reward += 1.5 * (self.previous_distance - distance_to_sphere)
        reward += -0.3 * abs(self.relative_azimuth)

        # Terminal conditions
        terminated = False
        truncated = False

        too_close_to_obstacles = False
        # for i in range(1, len(self.lidar_readings)):
        #     if self.lidar_readings[i] < 0.32 and self.lidar_readings[i] > 0.01:
        #         reward += -0.1/self.lidar_readings[i]
        #
        #     elif self.lidar_readings[i] <= 0.01:
        #         too_close_to_obstacles = True
        #         break

        self.episode_return += reward

        if contact_with_obstacles or too_close_to_obstacles:
            reward += -20.0
            self.episode_return += reward
            self.last_episode_result = "collision"
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = time.time() - self.episode_time_begin
            terminated = True

        elif distance_to_sphere < 2.0 or contact_with_sphere:
            reward += 200
            self.episode_return += reward
            self.last_episode_result = "success"
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = time.time() - self.episode_time_begin
            terminated = True

        elif self.current_step >= self.max_episode_steps:
            # reward = -50.0
            # self.episode_return += reward
            self.last_episode_result = "timeout"
            info["steps_taken"] = self.current_step
            info["episode_time_length"] = time.time() - self.episode_time_begin
            truncated = True

        # if np.abs(self.previous_distance - distance_to_sphere) < 0.01 and not self.training_mode:
        #     self.stuck_counter += 1
        #     if self.stuck_counter >= 100:
        #         reward = -10.0
        #         self.episode_return += reward
        #         self.last_episode_result = "timeout"
        #         info["steps_taken"] = self.current_step
        #         info["episode_time_length"] = time.time() - self.episode_time_begin
        #         terminated = True

        self.previous_distance = distance_to_sphere

        info["episode_result"] = self.last_episode_result

        # Render if needed
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()

            if not self.training_mode:  # During evaluation
                # Sleep exactly the model timestep to achieve real-time rendering
                # time.sleep(self.model.opt.timestep)
                time.sleep(0.01)  # Adjust this value for smoother rendering
                return True
            # elif self.current_step % 10 == 0:
            #     time.sleep(0.001)  # Faster rendering during training
            return True
        return False

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

