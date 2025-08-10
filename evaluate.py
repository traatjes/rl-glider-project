import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src import config

register(id=config.ENV_ID, entry_point='src.glider_env:GliderEnv')

try:
    eval_env = DummyVecEnv([lambda: gym.make(config.ENV_ID)])
    eval_env = VecNormalize.load(config.VEC_NORMALIZE_PATH, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
except FileNotFoundError:
    print("Could not find normalization stats. Please train the agent first.")
    exit()

try:
    model = PPO.load(config.MODEL_PATH, env=eval_env)
except FileNotFoundError:
    print(f"Could not find model. Please train the agent first.")
    exit()

# --- Run the evaluation for stats ---
all_episode_lengths = []
all_episode_rewards = []
all_max_distance = []

for _ in range(config.EVAL_EPISODES):
    obs = eval_env.reset()
    done = False
    truncated = False
    episode_length = 0
    episode_reward = 0
    episode_max_dist = 0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = eval_env.step(action)
        done = dones[0]
        truncated = info[0].get("TimeLimit.truncated", False)
        episode_length += 1
        episode_reward += reward[0]
        if done or truncated:
            episode_max_dist = info[0]['ep_max_x_distance']

    all_episode_rewards.append(episode_reward)
    all_max_distance.append(episode_max_dist)
    all_episode_lengths.append(episode_length)

print("--- Evaluation Results ---")
print(f"Average episode length: {np.mean(all_episode_lengths):.2f} steps")
print(f"Average episode reward: {np.mean(all_episode_rewards):.2f}")
print(f"Average max distance travelled: {np.mean(all_max_distance):.2f} meters")


# Plotting a single flight path
print("\n✈️  Visualizing one flight path...")
vis_env = gym.make(config.ENV_ID)
obs_vis, info = vis_env.reset()
done = False
truncated = False

path_x = [vis_env.unwrapped.position[0]]
path_y = [vis_env.unwrapped.position[1]]
path_alt = [vis_env.unwrapped.altitude]
max_x_visualization = vis_env.unwrapped.position[0]

while not (done or truncated):
    normalized_obs = eval_env.normalize_obs(obs_vis)
    action, _ = model.predict(normalized_obs, deterministic=True)
    obs_vis, _, terminated, truncated, info = vis_env.step(action)
    done = terminated or truncated
    path_x.append(vis_env.unwrapped.position[0])
    path_y.append(vis_env.unwrapped.position[1])
    path_alt.append(vis_env.unwrapped.altitude)
    max_x_visualization = max(max_x_visualization, vis_env.unwrapped.position[0])
print(f"Visualized flight stats: Final X = {path_x[-1]:.2f}m, Max X = {max_x_visualization:.2f}m")


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

# Plot 1: Flight Path
thermals = vis_env.unwrapped.thermals
for x, y, strength, radius in thermals:
    circle = plt.Circle((x, y), radius, color='r', alpha=0.2)
    ax1.add_artist(circle)
    ax1.text(x, y, 'T', ha='center', va='center', fontweight='bold', alpha=0.5)

points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(0, len(segments))
lc = LineCollection(segments, cmap='viridis', norm=norm)
lc.set_array(np.arange(len(segments)))
lc.set_linewidth(2)
line = ax1.add_collection(lc)

ax1.plot(path_x[0], path_y[0], 'go', markersize=10)  # Start
ax1.plot(path_x[-1], path_y[-1], 'ko', markersize=10)  # End

custom_lines = [
    Line2D([0], [0], color='g', marker='o', linestyle='', markersize=10, label='Start'),
    Line2D([0], [0], color='k', marker='o', linestyle='', markersize=10, label='End')
]
ax1.legend(handles=custom_lines)

ax1.set_title("Agent Flight Path (Colored by Time)")
ax1.set_xlabel("X Distance (m)")
ax1.set_ylabel("Y Position (m)")
ax1.grid(True)
ax1.set_aspect('equal')


# Plot 2: Altitude Profile
time_steps = range(len(path_alt))
ax2.plot(time_steps, path_alt, 'r-')
ax2.axhline(y=vis_env.unwrapped.start_altitude, color='g', linestyle='--', label='Start Altitude')
ax2.set_title("Altitude Profile")
ax2.set_xlabel("Time (steps)")
ax2.set_ylabel("Altitude (m)")
ax2.legend()
ax2.grid(True)
plt.show()



eval_env.close()
vis_env.close()
