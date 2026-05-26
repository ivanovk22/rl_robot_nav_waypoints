import os
import time
import xml.etree.ElementTree as ET
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

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

        obs = np.concatenate([self.lidar_readings, [distance, self.relative_azimuth]])


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
        distance_to_sphere = np.linalg.norm(sphere_pos - cube_pos)

        return {
            "distance_to_sphere": distance_to_sphere,
            "cube_position": cube_pos,
            "sphere_position": sphere_pos
        }



    def reset(self, seed=None, options=None):
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
        self.stuck_counter = 0

        # Define arena boundaries with buffer

        """
                                                                  x
                                                                  ^
        Robot's orientation in the reference frame of mujoco y<---|
        yaw = 0 → pointing along + X
        yaw = +π / 2 → pointing along + Y
        yaw = π → pointing along - X
        yaw = -π / 2 → pointing along - Y 
        """

        cube_x = 17.62
        cube_y = 20.306
        cube_yaw = np.pi/2

        self.waypoints = [
            np.array([17.6, 22.3, 2.0]),
            np.array([14.956, 22.618, 2.0]),
            # np.array([8.535, 22.633, 2.0]),
            # np.array([1.503, 22.633, 2.0]),
            # np.array([-5.893, 22.991, 2.0]),
            # np.array([-6.873, 19.245, 2.0]),
            # np.array([-6.873, 8.458, 2.0]),
            # np.array([-7.162, 3.347, 2.0]),
            # np.array([-6.584, 0.597, 2.0]),
            # np.array([-7.019, -2.881, 2.0]),
            # np.array([-6.838, -5.265, 2.0]),
            # np.array([-5.107, -6.143, 2.0]),
            # np.array([4.02, -8.118, 2.0]),
            # np.array([11.0, -8.134, 2.0]),
            # np.array([16.838, -6.525, 2.0]),
            # np.array([17.511, -4.222, 2.0]),
            # np.array([17.593, 4.008, 2.0]),
            # np.array([17.593, 12.719, 2.0]),
            # np.array([17.617, 20.306, 2.0]),
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
        self.previous_distance = info["distance_to_sphere"]

        return observation, info



    def step(self, action):
        # Movement parameters
        move_max_linspeed = 0.4
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
        theta_to_plot = (theta + np.pi) % (2*np.pi) - np.pi # wrap it
        self.theta_positions.append(theta_to_plot)
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
        reward_components = {}
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
        MAX_DIST_STEP = 0.25  # tune (typical robot movement per step)

        r_progress = 2.0 * np.clip((self.previous_distance - distance_to_sphere) / MAX_DIST_STEP, -1, 1)
        reward += r_progress
        reward_components["progress"] = r_progress        # Find the minimum lidar reading (assuming lidar_readings is a numpy array)

        # Find the minimum lidar reading (assuming lidar_readings is a numpy array)
        r_angle = -0.3 * (abs(self.relative_azimuth) / np.pi)
        reward += r_angle
        reward_components["angle"] = r_angle        # Negative reward as time passes, we want to reach the goal faster without stucking somewhere.
        # I hope that this will make the robot move if he is stuck
        r_time = -0.05  # Time penalty
        reward += r_time
        reward_components["time"] = r_time

        # ACTION SMOOTHNESS
        r_action = -0.03 * (action[0] ** 2 + action[1] ** 2)
        reward += r_action
        reward_components["action"] = r_action

        lin_change = abs(linear_velocity - prev_lin_vel)
        ang_change = abs(angular_velocity - prev_ang_vel)
        r_smooth = -0.40 * lin_change - 0.15 * ang_change
        reward += r_smooth
        reward_components["smoothness"] = r_smooth

        reward += -0.02 * abs(angular_velocity)
        reward_components["angular_bias"] = -0.02 * abs(angular_velocity)


        # Terminal conditions
        terminated = False
        truncated = False

        # LIDAR PENALTY
        r_lidar = 0
        too_close_to_obstacles = False
        d_min = 1
        for i in range(1, len(self.lidar_readings)):
            if self.lidar_readings[i] < 0.32 and self.lidar_readings[i] > 0.1:
                d = self.lidar_readings[i]
                if d_min > d:
                    d_min = d
            elif self.lidar_readings[i] <= 0.1:
                too_close_to_obstacles = True
                break
        if d_min != 1:
            r_lidar = -0.6 * np.exp(-d_min / 0.2)
        reward += r_lidar
        reward_components["lidar"] = r_lidar


        self.episode_return += reward

        if contact_with_obstacles or too_close_to_obstacles:
            r_collision = -10.0
            reward += r_collision
            reward_components["collision"] = r_collision

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

        elif distance_to_sphere < 2.0 or contact_with_sphere:
            self.current_waypoint_idx += 1

            # Check if final waypoint reached
            if self.current_waypoint_idx >= len(self.waypoints):
                r_success = 20.0
                reward += r_success
                reward_components["success"] = r_success

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
        info["reward_components"] = reward_components

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


