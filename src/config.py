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

# True:  [altitude, v_velocity, dist_to_thermal, angle_to_thermal, thermal_strength]
# False: [x, y, heading, altitude, v_velocity]
USE_AUGMENTED_OBSERVATIONS = True
VIEWING_RANGE = 1000.0 # 1km
MAX_OBSERVED_THERMALS = 5

# Glider initial state and physics
START_POS = np.array([0.0, 50.0], dtype=np.float32)
START_ALTITUDE = 300.0
START_HEADING = 0
GLIDER_SPEED = 10.0
SINK_RATE = 2.8
TURN_RATE = 15.0
DELTA_T = 1.0

# Thermal generation
EASY_MODE = False
NUM_THERMALS = 75
THERMAL_MIN_X, THERMAL_MAX_X = 200, 8000
THERMAL_MIN_Y, THERMAL_MAX_Y = -300, 300
EASY_THERMAL_MIN_STRENGTH, EASY_THERMAL_MAX_STRENGTH = 7.0, 9.0
EASY_THERMAL_MIN_RADIUS, EASY_THERMAL_MAX_RADIUS = 250, 350
THERMAL_MIN_STRENGTH, THERMAL_MAX_STRENGTH = 3.0, 6.0
THERMAL_MIN_RADIUS, THERMAL_MAX_RADIUS = 50, 90

# --- Agent & Training Parameters ---
# PPO Hyperparameters
LEARNING_RATE = 0.0001
N_STEPS = 8192
N_EPOCHS = 10
ENT_COEF = 0.05
GAMMA = 0.999
# Training settings
TOTAL_TIMESTEPS = 500000
MAX_EPISODE_STEPS = 2000

# --- Reward Function Parameters ---
REWARD_COEFF_FORWARD = 1.0
REWARD_COEFF_DISTANCE = 0.1
REWARD_COEFF_HEADING = 0.5   # Bonus for pointing at a thermal when cruising
REWARD_COEFF_UPDRAFT = 2.0   # Bonus for being in strong lift when climbing
REWARD_COEFF_DOLPHIN = 0.2    # Bonus for efficient cruising
REWARD_TERMINAL_PENALTY = -75
REWARD_COEFF_DISCOVERY = 50.0   # Large one-time bonus for finding a new thermal
REWARD_COEFF_CRUISING = 0.5    # Small bonus for gliding forward when high
REWARD_COEFF_MILESTONE = 100.0
CRUISING_ALTITUDE_THRESHOLD = 350.0 # Min altitude to get the cruising bonus
# Loitering Penalty Coefficients
LOITERING_PENALTY_THRESHOLD = 30  # Steps before penalty starts
REWARD_COEFF_LOITERING = 0.01      # Penalty strength

# --- Soaring Strategy Parameters ---
CLOUDBASE_ALTITUDE = 1200.0 # Altitude at which climbing rewards diminish to zero

# --- Evaluation Parameters ---
EVAL_EPISODES = 50
