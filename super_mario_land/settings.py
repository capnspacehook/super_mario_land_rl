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
GAME_AREA_EMBEDDING_DIM = 4
FEATURES_FC_HIDDEN_UNITS = 256
ENTITY_EMBEDDING_DIM = 4
LSTM_HIDDEN_UNITS = 512
ACTOR_HIDDEN_UNITS = 512
CRITIC_HIDDEN_UNITS = 512

# Reward values
# punishment for loosing powerup
HIT_PUNISHMENT = -10
DEATH_PUNISHMENT = -30
GAME_OVER_PUNISHMENT = -50
# added every step to encourage finishing levels faster
# TODO: remove this?
CLOCK_PUNISHMENT = -0.01
# score reward is this multiplied by the amount the score increased
SCORE_REWARD_COEF = 0.01
COIN_REWARD = 2  # +100 score when getting a coin must be factored in
PROGRESS_REWARD_COEF = 0
# main reward, mario's X speed multiplied by this
FORWARD_REWARD_COEF = 1
# punish going backwards, mario's X speed is multiplied by this
BACKWARD_PUNISHMENT_COEF = 0.25
MUSHROOM_REWARD = 15  # 1000 score
# TODO: add reward for killing enemies with fireballs
FLOWER_REWARD = 15  # 1000 score
# TODO: make reward when star bug is fixed/mitigated
STAR_REWARD = -35  # 1000 score
HEART_REWARD = 30  # 1000 score
# if life or heart farming is detected (getting another
# life, dying, respawning and repeat forever) issue a punishment
HEART_FARM_PUNISHMENT = -60
# reward standing on a moving platform when it's getting closer
# to another moving platform
MOVING_PLATFORM_DISTANCE_REWARD_MAX = 0.5
# reward standing on a moving platform while it's moving forward
# TODO: only reward when mario isn't moving
MOVING_PLATFORM_X_REWARD_COEF = 0.15
# reward standing on a moving platform while it's rising
MOVING_PLATFORM_Y_REWARD_COEF = 1.25
# reward standing on a boulder in world 3 which is necessary to cross
# certain large gaps in some levels and is difficult to learn otherwise
BOULDER_REWARD = 5
HIT_BOSS_REWARD = 5
KILL_BOSS_REWARD = 25
LEVEL_CLEAR_REWARD = 35
# reward clearing through top more
LEVEL_CLEAR_TOP_REWARD = 20
# increase reward by amount of lives left, powerups active
LEVEL_CLEAR_LIVES_COEF_REWARD = 5
LEVEL_CLEAR_BIG_REWARD = 5
LEVEL_CLEAR_FIRE_REWARD = 10

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
FRAME_CELL_CHECK = 120
ENTITIES_IGNORE_Y_POS = [FLYING_MOTH_ARROW_TYPE_ID, BONE_FISH_TYPE_ID, SEAHORSE_TYPE_ID]
