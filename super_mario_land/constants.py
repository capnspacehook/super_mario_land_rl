GAME_AREA_OBS = "gameArea"
MARIO_INFO_OBS = "marioInfo"
ENTITY_ID_OBS = "entityID"
ENTITY_INFO_OBS = "entityInfo"
SCALAR_OBS = "scalar"

MARIO_INFO_SIZE = 7
ENTITY_INFO_SIZE = 10
SCALAR_SIZE = 10

MARIO_MAX_X_SPEED = 5
MARIO_MAX_Y_SPEED = 4
ENTITY_MAX_RAW_X_SPEED = 10
ENTITY_MAX_RAW_Y_SPEED = 12
ENTITY_MAX_MEAN_X_SPEED = 3
ENTITY_MAX_MEAN_Y_SPEED = 3
# distance between 0, 0 and 160, 210 is 264.007... so rounding up
MAX_EUCLIDEAN_DISTANCE = 265

ON_GROUND_FRAMES = 3

POWERUP_STATUSES = 4
MAX_INVINCIBILITY_TIME = 960

MAX_ENTITY_ID = 33

# max number of objects that can be on screen at once (excluding mario)
N_ENTITIES = 10

MAX_REL_X_POS = 160
MAX_Y_POS = 210
MAX_X_DISTANCE = 200
MAX_Y_DISTANCE = 200

# game area dimensions
GAME_AREA_HEIGHT = 16
GAME_AREA_WIDTH = 20

# update if the maximum tile value changes
MAX_TILE = 44

LEVEL_END_X_POS = {
    "1-1": 2600,
    "1-2": 2440,
    "1-3": 2588,
    "2-1": 2760,
    "2-2": 2440,
    "3-1": 3880,
    "3-2": 2760,
    "3-3": 2588,
    "4-1": 3880,
    "4-2": 3400,
}
