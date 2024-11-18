from torch import nn

from super_mario_land.ram import (
    STATUS_SMALL,
    STATUS_BIG,
    STATUS_FIRE,
    FLYING_MOTH_ARROW_TYPE_ID,
    BONE_FISH_TYPE_ID,
    SEAHORSE_TYPE_ID,
)

# General env settings
FRAME_SKIP = 4

# Observation settings
N_STATE_STACK = 6  # number of games states to use to calculate mean speeds

# Policy settings
ACTIVATION_FN = nn.ReLU

# Env stochasticity settings
RANDOM_NOOP_FRAMES = 60
RANDOM_NOOP_FRAMES_WITH_ENEMIES = 20
RANDOM_POWERUP_CHANCE = 25
STARTING_LIVES_MIN = 1
STARTING_LIVES_MAX = 3
MIN_RANDOM_TIME = 60

# Eval settings
START_LEVEL = "1-1"
DEFAULT_LIVES_LEFT = 2
DEFAULT_COINS = 0
DEFAULT_POWERUP = STATUS_SMALL

# Time settings
MIN_TIME = 10
STARTING_TIME = 400
# amount of time to subtract when mario respawns
# the base game always sets the timer to the max
# time 400 when respawning but subtracting time
# upon death this way prevents farming lives
# and an endless episode
DEATH_TIME_LOSS = 10

# Heart farming detection settings
HEART_FARM_X_POS_MULTIPLE = 15

# Cell selection settings
MAX_START_LEVEL = "1-1"
X_POS_MULTIPLE = 150
Y_POS_MULTIPLE = 30
ENTITY_X_POS_MULTIPLE = 200
ENTITY_Y_POS_MULTIPLE = 40
ENEMY_SAFE_DISTANCE = 50
FRAME_CELL_CHECK = 120
ENTITIES_IGNORE_Y_POS = [FLYING_MOTH_ARROW_TYPE_ID, BONE_FISH_TYPE_ID, SEAHORSE_TYPE_ID]
