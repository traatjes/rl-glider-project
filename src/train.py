# In train.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from src import agent, config
from src.callback import FinalLoggerCallback
import time

print("🚀 Starting training process...")

# 1. Register the environment
register(id=config.ENV_ID, entry_point='src.glider_env:GliderEnv')

# 2. Define the keywords for VecMonitor
custom_info_keywords = ("ep_avg_altitude", "ep_max_x_distance", "ep_time_in_thermal")

# 3. Create and wrap the environment
base_env = DummyVecEnv([lambda: gym.make(config.ENV_ID)])
base_env = VecMonitor(base_env,
                      filename=str(config.LOGS_PATH / "monitor.csv"),
                      info_keywords=custom_info_keywords)

# 4. Load or create VecNormalize
if config.VEC_NORMALIZE_PATH.exists():
    env = VecNormalize.load(config.VEC_NORMALIZE_PATH, base_env)
    env.training = True
else:
    env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.)

# 5. Create or load the agent
if config.MODEL_PATH.exists():
    print(f"Loading model from {config.MODEL_PATH}")
    glider_agent = agent.load_agent(config.MODEL_PATH)
    glider_agent.set_env(env)
else:
    print("No pre-trained model found, starting from scratch.")
    glider_agent = agent.create_agent(env)

# 6. Train the agent
print(f"Training the agent. (Press Ctrl+C to stop and save)")
callback = FinalLoggerCallback()

try:
    while True:
        glider_agent.learn(total_timesteps=config.TOTAL_TIMESTEPS,
                           progress_bar=True,
                           reset_num_timesteps=False,
                           callback=callback)

        print(f"Training segment complete. Saving progress...")
        config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
        env.save(config.VEC_NORMALIZE_PATH)
        agent.save_agent(glider_agent)
        print("Starting next segment...")
        time.sleep(5)

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")

finally:
    print("Saving final model and environment stats...")
    config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    env.save(config.VEC_NORMALIZE_PATH)
    agent.save_agent(glider_agent)
    env.close()

print(f"Training complete. Final model saved to:\n{config.MODEL_PATH}")