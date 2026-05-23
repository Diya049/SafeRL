# ─────────────────────────────────────────────
#  Grid World Config
# ─────────────────────────────────────────────
GRID_SIZE = 13

# Rewards
STEP_COST   = -1.0
GOAL_REWARD = 250.0

# Training
NUM_EPISODES   = 5000
MAX_STEPS      = 150
EVAL_EVERY     = 50
LOG_BEST       = True

# Q-learning base
ALPHA        = 0.05
GAMMA        = 0.99
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.995

# ─── Lagrangian RL (Method 1) ──────────────────
LAG_COST_LIMIT       = 0.1   # max allowed avg constraint violation per step
LAG_LAMBDA_LR        = 0.05  # learning rate for the dual variable λ
LAG_LAMBDA_INIT      = 0.0
LAG_LAMBDA_MAX       = 10.0

# ─── Safety Layer (Method 2) ──────────────────
SL_PROX_RADIUS       = 2     # Manhattan distance at which safety layer fires
SL_PENALTY_IN_TRAIN  = 30    # extra penalty applied during SL training
