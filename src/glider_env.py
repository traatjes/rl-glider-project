import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src import config


class GliderEnv(gym.Env):
    """Custom Environment for a Glider that follows the gymnasium API."""
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)

        if config.USE_AUGMENTED_OBSERVATIONS:
            # Obs: [altitude, v_velocity] + N * [normalized_dist, sin, cos, strength]
            glider_obs_low = [0, -np.inf]
            glider_obs_high = [np.inf, np.inf]

            # [dist, sin, cos, str]
            thermal_low_bounds = [0.0, -1.0, -1.0, 0.0]
            thermal_high_bounds = [1.0, 1.0, 1.0, np.inf]

            thermals_obs_low = []
            thermals_obs_high = []
            for _ in range(config.MAX_OBSERVED_THERMALS):
                thermals_obs_low.extend(thermal_low_bounds)
                thermals_obs_high.extend(thermal_high_bounds)

            low = np.array(glider_obs_low + thermals_obs_low, dtype=np.float32)
            high = np.array(glider_obs_high + thermals_obs_high, dtype=np.float32)

            self.observation_space = spaces.Box(low, high, dtype=np.float32)
        else:
            # Obs: [x, y, heading, altitude, v_velocity]
            low = np.array([-np.inf, -np.inf, 0, 0, -np.inf], dtype=np.float32)
            high = np.array([np.inf, np.inf, 360, np.inf, np.inf], dtype=np.float32)
            self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.start_pos = config.START_POS
        self.start_altitude = config.START_ALTITUDE
        self.start_heading = config.START_HEADING
        self.glider_speed = config.GLIDER_SPEED
        self.sink_rate = config.SINK_RATE
        self.turn_rate = config.TURN_RATE
        self.delta_t = config.DELTA_T
        self.max_steps = config.MAX_EPISODE_STEPS

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = self.start_pos.copy()
        self.altitude = self.start_altitude
        self.heading = self.start_heading
        self.vertical_velocity = 0
        self.thermals = []
        self.next_milestone = 1000.0

        # Episode stats
        self.last_x = self.position[0]
        self.max_x = self.position[0]
        self.episode_altitude_sum = 0
        self.episode_steps_in_thermal = 0
        self.in_thermal = False
        self.updraft = 0.0
        self.visited_thermals = set()  # ✨ Add a set to track visited thermals

        for _ in range(config.NUM_THERMALS):
            if config.EASY_MODE:
                strength = self.np_random.uniform(config.EASY_THERMAL_MIN_STRENGTH, config.EASY_THERMAL_MAX_STRENGTH)
                radius = self.np_random.uniform(config.EASY_THERMAL_MIN_RADIUS, config.EASY_THERMAL_MAX_RADIUS)
            else:
                strength = self.np_random.uniform(config.THERMAL_MIN_STRENGTH, config.THERMAL_MAX_STRENGTH)
                radius = self.np_random.uniform(config.THERMAL_MIN_RADIUS, config.THERMAL_MAX_RADIUS)
            x = self.np_random.uniform(config.THERMAL_MIN_X, config.THERMAL_MAX_X)
            y = self.np_random.uniform(config.THERMAL_MIN_Y, config.THERMAL_MAX_Y)
            self.thermals.append([x, y, strength, radius])

        thermals_array = np.array(self.thermals)
        self.thermal_positions = thermals_array[:, :2]
        self.thermal_strengths = thermals_array[:, 2]
        self.thermal_radii = thermals_array[:, 3]

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1

        # 1. Update Glider State
        if action == 0: self.heading -= self.turn_rate
        elif action == 2: self.heading += self.turn_rate
        self.heading %= 360
        heading_rad = np.deg2rad(self.heading)
        self.position[0] += self.glider_speed * np.cos(heading_rad) * self.delta_t
        self.position[1] += self.glider_speed * np.sin(heading_rad) * self.delta_t

        # 2. Update Altitude (your logic is perfect)
        dists = np.linalg.norm(self.position - self.thermal_positions, axis=1)
        inside_mask = dists < self.thermal_radii
        updraft = np.sum(self.thermal_strengths[inside_mask])
        self.vertical_velocity = updraft - self.sink_rate
        self.altitude += self.vertical_velocity * self.delta_t

        # 3. Check for Termination Conditions
        terminated = self.altitude <= 0
        truncated = self.current_step >= self.max_steps

        # 4. Calculate Reward
        reward = 0.0

        # Check which thermals the agent is currently inside
        current_dists = np.linalg.norm(self.position - self.thermal_positions, axis=1)
        inside_indices = np.where(current_dists < self.thermal_radii)[0]

        if len(inside_indices) > 0:
            # --- Inside a Thermal ---
            for idx in inside_indices:
                if idx not in self.visited_thermals:
                    # Give a one-time bonus for discovering a new thermal
                    reward += config.REWARD_COEFF_DISCOVERY
                    self.visited_thermals.add(idx)
        else:
            # --- Cruising (Not in a Thermal) ---
            if self.altitude > config.CRUISING_ALTITUDE_THRESHOLD:
                progress = self.position[0] - self.last_x
                if progress > 0:  # Only reward positive progress
                    reward += config.REWARD_COEFF_CRUISING * progress

        if self.position[0] >= self.next_milestone:
            reward += config.REWARD_COEFF_MILESTONE
            self.next_milestone += 1000.0  # Set the next milestone

        self.last_x = self.position[0]

        if terminated or truncated:
            reward += config.REWARD_COEFF_DISTANCE * self.position[0]
            if terminated:
                reward += config.REWARD_TERMINAL_PENALTY


        # 5. Update Info and Stats
        self.in_thermal = len(inside_indices) > 0
        self.max_x = max(self.max_x, self.position[0])
        self.episode_altitude_sum += self.altitude
        if self.in_thermal:
            self.episode_steps_in_thermal += 1

        obs = self._get_obs()
        info = self._get_info()

        if terminated or truncated:
            total_steps = self.current_step
            info['ep_avg_altitude'] = self.episode_altitude_sum / total_steps
            info['ep_max_x_distance'] = self.max_x - self.start_pos[0]
            info['ep_time_in_thermal'] = 100 * self.episode_steps_in_thermal / total_steps

        return obs, reward, terminated, truncated, info


    def _get_closest_thermal_info(self):
        """Helper to get distance, angle, and strength of the nearest thermal."""
        if len(self.thermals) == 0:
            return None

        dists = np.linalg.norm(self.position - self.thermal_positions, axis=1)
        nearest_idx = np.argmin(dists)

        vec_to_thermal = self.thermal_positions[nearest_idx] - self.position
        dist_to_thermal = dists[nearest_idx]

        angle_to_thermal_abs = np.rad2deg(np.arctan2(vec_to_thermal[1], vec_to_thermal[0]))
        angle_relative = angle_to_thermal_abs - self.heading
        angle_relative = (angle_relative + 180) % 360 - 180

        strength = self.thermal_strengths[nearest_idx]

        return dist_to_thermal, angle_relative, strength

    def _get_obs(self):
        if not config.USE_AUGMENTED_OBSERVATIONS:
            return np.array([self.position[0], self.position[1], self.heading, self.altitude, self.vertical_velocity], dtype=np.float32)

        # Base observation: the glider's own state
        obs_list = [self.altitude, self.vertical_velocity]

        # --- Find all thermals within the viewing range ---
        vecs_to_thermals = self.thermal_positions - self.position
        dists = np.linalg.norm(vecs_to_thermals, axis=1)
        visible_indices = np.where(dists < config.VIEWING_RANGE)[0]
        sorted_indices = visible_indices[np.argsort(dists[visible_indices])] # sort by closest first

        # --- Populate the observation vector ---
        thermals_observed = 0
        for i in range(config.MAX_OBSERVED_THERMALS):
            if i < len(sorted_indices):
                idx = sorted_indices[i]

                angle_abs = np.rad2deg(np.arctan2(vecs_to_thermals[idx, 1], vecs_to_thermals[idx, 0]))
                angle_rel = (angle_abs - self.heading + 180) % 360 - 180
                angle_rel_rad = np.deg2rad(angle_rel)

                # Normalizing the distance by the viewing range so it's always between 0 and 1.
                normalized_distance = 1.0 - (dists[idx] / config.VIEWING_RANGE)

                obs_list.extend([
                    normalized_distance,
                    np.sin(angle_rel_rad),
                    np.cos(angle_rel_rad),
                    self.thermal_strengths[idx]
                ])
                thermals_observed += 1
            else:
                # If there are no more thermals to see, pad with 0
                obs_list.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(obs_list, dtype=np.float32)

    def _get_info(self):
        return {
            "altitude": self.altitude,
            "heading": self.heading,
            "position_x": self.position[0],
            "position_y": self.position[1],
            "vertical_velocity": self.vertical_velocity
        }