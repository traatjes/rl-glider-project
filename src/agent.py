from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from src import config


def create_agent(env):
    """Creates a PPO agent with hyperparameters from the config file."""
    model = RecurrentPPO(
        policy='MlpLstmPolicy',
        env=env,
        verbose=1,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        n_epochs=config.N_EPOCHS,
        ent_coef=config.ENT_COEF,
        gamma=config.GAMMA,
        tensorboard_log=config.LOGS_PATH
    )
    return model

def train_agent(agent, total_timesteps=config.TOTAL_TIMESTEPS, callback=None):
    """Trains the agent for a given number of timesteps."""
    agent.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
    return agent

def save_agent(agent, file_path=config.MODEL_PATH):
    """Saves the trained agent's model."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(file_path)
    print(f"✅ Agent saved to {file_path}")

def load_agent(file_path=config.MODEL_PATH):
    """Loads a pre-trained agent's model."""
    model = PPO.load(file_path)
    return model