from stable_baselines3 import PPO
from src import config


def create_agent(env):
    """Creates a PPO agent with hyperparameters from the config file."""
    model = PPO(
        policy=config.POLICY,
        env=env,
        verbose=1,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        n_epochs=config.N_EPOCHS,
        ent_coef=config.ENT_COEF,
        gamma=config.GAMMA,
        tensorboard_log=config.LOGS_PATH,
        clip_range=0.1,
        gae_lambda=0.9
    )
    return model

def save_agent(agent, file_path=config.MODEL_PATH):
    """Saves the trained agent's model."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(file_path)
    print(f"✅ Agent saved to {file_path}")

def load_agent(file_path=config.MODEL_PATH):
    """Loads a pre-trained agent's model."""
    model = PPO.load(file_path)
    return model