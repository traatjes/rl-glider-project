import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src import config

print(f"--- Running Evaluation with DIFFICULTY = {config.DIFFICULTY} ---")
register(id=config.ENV_ID, entry_point='src.glider_env:GliderEnv')
try:
    eval_env = DummyVecEnv([lambda: gym.make(config.ENV_ID)])
    eval_env = VecNormalize.load(config.VEC_NORMALIZE_PATH, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    model = PPO.load(config.MODEL_PATH, env=eval_env)
except FileNotFoundError:
    print("Could not find model or normalization stats. Please train the agent first.")
    exit()


def run_evaluation_batch(env, is_deterministic, num_episodes):
    """Runs a batch of episodes on a standard, re-randomizing environment."""
    print(f"\nRunning {num_episodes} episodes (deterministic={is_deterministic}, seed=Random)...")
    batch_distances = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = truncated = False
        episode_max_dist = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=is_deterministic)
            obs, _, dones, info = env.step(action)
            done = dones[0]
            truncated = info[0].get("TimeLimit.truncated", False)
            if done or truncated:
                episode_max_dist = info[0].get('ep_max_x_distance', 0)
        batch_distances.append(episode_max_dist)
    return batch_distances


all_max_distance_stochastic = run_evaluation_batch(eval_env, is_deterministic=False, num_episodes=config.EVAL_EPISODES)
all_max_distance_deterministic = run_evaluation_batch(eval_env, is_deterministic=True,
                                                      num_episodes=config.EVAL_EPISODES)



def run_fixed_map_batch(vec_normalize_env, num_episodes, fixed_seed):
    """Runs a batch of episodes on a SINGLE, NON-RANDOM map."""
    print(f"\nRunning {num_episodes} episodes (deterministic=False, seed={fixed_seed})...")

    fixed_env = gym.make(config.ENV_ID)
    batch_distances = []
    for _ in range(num_episodes):
        obs, _ = fixed_env.reset(seed=fixed_seed)
        done = truncated = False
        while not (done or truncated):
            normalized_obs = vec_normalize_env.normalize_obs(obs)
            action, _ = model.predict(normalized_obs, deterministic=False)
            obs, _, terminated, truncated, info = fixed_env.step(action)
            done = terminated or truncated
        batch_distances.append(info.get('ep_max_x_distance', 0))
    fixed_env.close()
    return batch_distances


all_max_distance_fixed_map = run_fixed_map_batch(eval_env, num_episodes=config.EVAL_EPISODES, fixed_seed=123)

print("\n--- Evaluation Results ---")
print(f"Mean Distance (Random Maps, Stochastic Policy): {np.mean(all_max_distance_stochastic):.2f} m")
print(f"Mean Distance (Random Maps, Deterministic Policy): {np.mean(all_max_distance_deterministic):.2f} m")
print(f"Mean Distance (Fixed Map, Stochastic Policy): {np.mean(all_max_distance_fixed_map):.2f} m")
print(f"\nStandard Deviation (Random Maps, Stochastic Policy): {np.std(all_max_distance_stochastic):.2f} m")
print(f"Standard Deviation (Fixed Map, Stochastic Policy): {np.std(all_max_distance_fixed_map):.2f} m")



print("\nVisualizing one flight path...")
vis_env = gym.make(config.ENV_ID)
obs_vis, info = vis_env.reset()
done = False
truncated = False

path_x = [vis_env.unwrapped.position[0]]
path_y = [vis_env.unwrapped.position[1]]
path_alt = [vis_env.unwrapped.altitude]
max_x_visualization = vis_env.unwrapped.position[0]
episode_reward_visualization = 0

while not (done or truncated):
    normalized_obs = eval_env.normalize_obs(obs_vis)
    action, _ = model.predict(normalized_obs, deterministic=True)
    obs_vis, reward_vis, terminated, truncated, info = vis_env.step(action)
    done = terminated or truncated
    path_x.append(vis_env.unwrapped.position[0])
    path_y.append(vis_env.unwrapped.position[1])
    path_alt.append(vis_env.unwrapped.altitude)
    episode_reward_visualization += reward_vis

    max_x_visualization = max(max_x_visualization, vis_env.unwrapped.position[0])
print(f"Visualized flight stats: Final X = {path_x[-1]:.2f}m, Max X = {max_x_visualization:.2f}m, Reward = {episode_reward_visualization:.2f}")


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

# Plot 3: Histograms
fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
ax_hist.hist(all_max_distance_fixed_map, bins=20, edgecolor='black')
ax_hist.set_title("Distribution of Final Distances on a FIXED Map (Stochastic Policy)")

fig_hist, ax_hist = plt.subplots(figsize=(10, 6))

ax_hist.hist(all_max_distance_stochastic, bins=20, edgecolor='black', alpha=0.7, label='Stochastic (deterministic=False)')
ax_hist.hist(all_max_distance_deterministic, bins=20, edgecolor='black', alpha=0.7, label='Deterministic (deterministic=True)')

ax_hist.set_title("Distribution of Final Distances")
ax_hist.set_xlabel("Max Distance Achieved (m)")
ax_hist.set_ylabel("Number of Episodes")
ax_hist.grid(axis='y', alpha=0.75)
ax_hist.legend()

mean_sto = np.mean(all_max_distance_stochastic)
mean_det = np.mean(all_max_distance_deterministic)
ax_hist.axvline(mean_sto, color='blue', linestyle='dashed', linewidth=2, label=f'Stochastic Mean: {mean_sto:.0f}m')
ax_hist.axvline(mean_det, color='red', linestyle='dashed', linewidth=2, label=f'Deterministic Mean: {mean_det:.0f}m')

ax_hist.legend()

plt.tight_layout()
plt.show()

eval_env.close()
