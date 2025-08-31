"""
Central configuration file for the RL Glider project.
"""
import numpy as np
from pathlib import Path

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_PATH = RESULTS_DIR / "logs"
MODELS_PATH = RESULTS_DIR / "models"
MODEL_PATH = MODELS_PATH / "glider_agent.zip"
VEC_NORMALIZE_PATH = MODELS_PATH / "vec_normalize.pkl"

# --- Environment Parameters ---
ENV_ID = 'Glider-v0'

# Glider observation parameters
VIEWING_RANGE = 1000.0 # 1km
MAX_OBSERVED_THERMALS = 5

# Glider initial state and physics
START_POS = np.array([0.0, 0.0], dtype=np.float32)
START_ALTITUDE = 300.0
START_HEADING = 0
GLIDER_SPEED = 10.0
SINK_RATE = 2.8
TURN_RATE = 15.0
DELTA_T = 1.0

# Thermal generation
DIFFICULTY = 1.0
THERMAL_MIN_Y, THERMAL_MAX_Y = -1000, 1000
EASY_THERMAL_MIN_STRENGTH, EASY_THERMAL_MAX_STRENGTH = 5.0, 7.0
EASY_THERMAL_MIN_RADIUS, EASY_THERMAL_MAX_RADIUS = 200, 300
THERMAL_MIN_STRENGTH, THERMAL_MAX_STRENGTH = 3.0, 6.0
THERMAL_MIN_RADIUS, THERMAL_MAX_RADIUS = 50, 90
NUM_ZONES = 25  # We will create 25 zones
ZONE_SIZE = 1000 # Each zone is 1km long
THERMALS_PER_ZONE = 8 # Place 8 thermals in each zone (25*8 = 200 total)

# --- Agent & Training Parameters ---
# PPO Hyperparameters
POLICY = "MlpPolicy"
LEARNING_RATE = 0.00003
N_STEPS = 8192
N_EPOCHS = 10
ENT_COEF = 0.05
GAMMA = 0.999
# Training settings
TOTAL_TIMESTEPS = 500000
MAX_EPISODE_STEPS = 2000

# --- Reward Function Parameters ---
REWARD_COEFF_DISTANCE = 0.1
REWARD_COEFF_PROGRESS = 0.1
REWARD_COEFF_PROGRESS_NEG = 0.2
REWARD_TERMINAL_PENALTY = -75
REWARD_COEFF_MILESTONE = 100.0
REWARD_COEFF_ALTITUDE = 0.0004

# --- Evaluation Parameters ---
EVAL_EPISODES = 1000
