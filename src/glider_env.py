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

        # Obs: [altitude, v_velocity] + N * [normalized_dist, sin, cos, strength]
        glider_obs_low = [0, -np.inf]
        glider_obs_high = [np.inf, np.inf]
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


        self.start_pos = config.START_POS
        self.start_altitude = config.START_ALTITUDE
        self.start_heading = config.START_HEADING
        self.glider_speed = config.GLIDER_SPEED
        self.sink_rate = config.SINK_RATE
        self.turn_rate = config.TURN_RATE
        self.delta_t = config.DELTA_T
        self.max_steps = config.MAX_EPISODE_STEPS

    def _lerp(self, val1, val2, t):
        """Helper function for linear interpolation."""
        return val1 * (1 - t) + val2 * t

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = self.start_pos.copy()
        self.altitude = self.start_altitude
        self.heading = self.start_heading
        self.vertical_velocity = 0
        self.thermals = []
        self.next_milestone = 1000.0
        self.last_x = self.position[0]
        self.max_x = self.position[0]
        self.episode_altitude_sum = 0
        self.episode_steps_in_thermal = 0
        self.in_thermal = False
        self.updraft = 0.0
        self.visited_thermals = set()  # ✨ Add a set to track visited thermals

        for i in range(config.NUM_ZONES):
            # Define the start and end of the current zone
            zone_start_x = i * config.ZONE_SIZE
            zone_end_x = (i + 1) * config.ZONE_SIZE

            for _ in range(config.THERMALS_PER_ZONE):
                # Interpolate strength and radius based on the difficulty setting
                min_strength = self._lerp(config.EASY_THERMAL_MIN_STRENGTH, config.THERMAL_MIN_STRENGTH, config.DIFFICULTY)
                max_strength = self._lerp(config.EASY_THERMAL_MAX_STRENGTH, config.THERMAL_MAX_STRENGTH, config.DIFFICULTY)
                min_radius = self._lerp(config.EASY_THERMAL_MIN_RADIUS, config.THERMAL_MIN_RADIUS, config.DIFFICULTY)
                max_radius = self._lerp(config.EASY_THERMAL_MAX_RADIUS, config.THERMAL_MAX_RADIUS, config.DIFFICULTY)

                # Generate the thermal with the calculated properties *within the current zone*
                strength = self.np_random.uniform(min_strength, max_strength)
                radius = self.np_random.uniform(min_radius, max_radius)
                x = self.np_random.uniform(zone_start_x, zone_end_x)
                y = self.np_random.uniform(config.THERMAL_MIN_Y, config.THERMAL_MAX_Y)
                self.thermals.append([x, y, strength, radius])

        thermals_array = np.array(self.thermals)
        self.thermal_positions = thermals_array[:, :2]
        self.thermal_strengths = thermals_array[:, 2]
        self.thermal_radii = thermals_array[:, 3]

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1

        self._update_glider_state(action)
        terminated, truncated = self._check_termination()
        reward = self._calculate_reward(terminated, truncated)
        obs = self._get_obs()
        info = self._get_info()
        self._update_info_log(info, terminated, truncated)

        return obs, reward, terminated, truncated, info

    def _update_glider_state(self, action):
        # Update Glider State
        if action == 0: self.heading -= self.turn_rate
        elif action == 2: self.heading += self.turn_rate
        self.heading %= 360
        heading_rad = np.deg2rad(self.heading)
        self.position[0] += self.glider_speed * np.cos(heading_rad) * self.delta_t
        self.position[1] += self.glider_speed * np.sin(heading_rad) * self.delta_t

        # Update Altitude
        dists = np.linalg.norm(self.position - self.thermal_positions, axis=1)
        self.inside_mask = dists < self.thermal_radii
        updraft = np.sum(self.thermal_strengths[self.inside_mask])
        self.vertical_velocity = updraft - self.sink_rate
        self.altitude += self.vertical_velocity * self.delta_t

    def _check_termination(self):
        crashed = self.altitude <= 0
        over_ceiling = self.altitude >= 2000
        truncated = self.current_step >= self.max_steps
        terminated = crashed or over_ceiling

        return terminated, truncated

    def _calculate_reward(self, terminated, truncated):
        reward = 0.0

        # Unconditional Progress Reward/Penalty
        progress = self.position[0] - self.last_x
        if progress > 0:
            reward += config.REWARD_COEFF_PROGRESS * progress
        elif progress < 0 and not self.in_thermal:
            reward += config.REWARD_COEFF_PROGRESS_NEG * progress
        self.last_x = self.position[0]

        # Milestone Bonus
        if self.position[0] >= self.next_milestone:
            reward += config.REWARD_COEFF_MILESTONE
            self.next_milestone += 1000.0

        # Potential Energy Bonus
        reward += config.REWARD_COEFF_ALTITUDE * self.altitude

        # Termination Penalty
        if terminated or truncated:
            reward += config.REWARD_COEFF_DISTANCE * self.position[0]
            if terminated:
                reward += config.REWARD_TERMINAL_PENALTY

        return reward

    def _update_info_log(self, info, terminated, truncated):
        self.in_thermal = np.any(self.inside_mask)
        self.max_x = max(self.max_x, self.position[0])
        self.episode_altitude_sum += self.altitude
        if self.in_thermal:
            self.episode_steps_in_thermal += 1

        if terminated or truncated:
            total_steps = self.current_step
            info['ep_avg_altitude'] = self.episode_altitude_sum / total_steps
            info['ep_max_x_distance'] = self.max_x - self.start_pos[0]
            info['ep_time_in_thermal'] = 100 * self.episode_steps_in_thermal / total_steps


    def _get_obs(self):
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