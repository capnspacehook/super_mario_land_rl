import math
from typing import Any, Deque, Dict, List, Tuple

import numpy as np
from pyboy import PyBoy

from super_mario_land.constants import *
from super_mario_land.game_area import getGameArea
from super_mario_land.ram import MarioLandGameState, MarioLandObject
from super_mario_land.settings import *


def getObservation(
    pyboy: PyBoy,
    states: Deque[MarioLandGameState],
    onGround: bool,
) -> Dict[str, Any]:
    gameArea = getGameArea(pyboy, states[-1])
    marioInfo, entityIDs, entityInfos = getEntityIDsAndInfo(states)
    scalar = getScalarFeatures(states[-1], onGround)

    return {
        GAME_AREA_OBS: gameArea,
        MARIO_INFO_OBS: marioInfo,
        ENTITY_ID_OBS: entityIDs,
        ENTITY_INFO_OBS: entityInfos,
        SCALAR_OBS: scalar,
    }


def getEntityIDsAndInfo(
    states: Deque[MarioLandGameState],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prevState = states[-2]
    curState = states[-1]

    # a level was completed on the last step, discard the last step's
    # state to avoid incorrect speed and acceleration calculations
    if prevState.world != curState.world:
        prevState = curState

    # In rare occasions (jumping on a moving platform in 3-3 for
    # instance) the scroll X value is ~16, while in the next frame
    # the scroll X value is where it was before but the level block
    # increased making the X position wildly jump again. Until I
    # figure out how to properly fix this, clip the X (and Y for
    # good measure) speed to ensure it's always a sane value.
    curState.rawXSpeed = np.clip(curState.xPos - prevState.xPos, -MARIO_MAX_X_SPEED, MARIO_MAX_X_SPEED)
    curState.rawYSpeed = np.clip(curState.yPos - prevState.yPos, -MARIO_MAX_Y_SPEED, MARIO_MAX_Y_SPEED)

    # only consider speeds in states of the same life or level as the
    # current state
    resetIdx = -1
    for i, s in enumerate(states):
        if s.posReset:
            resetIdx = i
            break

    # don't calculate mean speed if the latest state was the reset one
    if resetIdx != N_STATE_STACK - 1:
        curState.meanXSpeed = np.mean([states[i].rawXSpeed for i in range(resetIdx + 1, N_STATE_STACK)])
        curState.meanYSpeed = np.mean([states[i].rawYSpeed for i in range(resetIdx + 1, N_STATE_STACK)])
    curState.xAccel = curState.meanXSpeed - prevState.meanXSpeed
    curState.yAccel = curState.meanYSpeed - prevState.meanYSpeed

    marioInfo = np.array(
        [
            scaledEncoding(curState.relXPos, MAX_REL_X_POS, True),
            scaledEncoding(curState.yPos, MAX_Y_POS, True),
            scaledEncoding(curState.meanXSpeed, MARIO_MAX_X_SPEED, False),
            scaledEncoding(curState.meanYSpeed, MARIO_MAX_Y_SPEED, False),
            scaledEncoding(curState.xAccel, MARIO_MAX_X_SPEED, False),
            scaledEncoding(curState.yAccel, MARIO_MAX_Y_SPEED, False),
            scaledEncoding(math.atan2(curState.meanXSpeed, curState.meanYSpeed), math.pi, False),
        ],
        dtype=np.float32,
    )
    marioPos = np.array((curState.xPos, curState.yPos))

    entities = []
    for obj in curState.objects:
        # attempt to find the same object in the previous frame's state
        # so the speed and acceleration can be calculated
        if len(prevState.objects) != 0:
            prevObj = findObjectInPrevState(obj, prevState)
            if prevObj is not None:
                obj.rawXSpeed = obj.xPos - prevObj.xPos
                obj.rawYSpeed = obj.yPos - prevObj.yPos
                rawXSpeeds = [prevObj.rawXSpeed, obj.rawXSpeed]
                rawYSpeeds = [prevObj.rawYSpeed, obj.rawYSpeed]
                obj.meanXSpeed, obj.meanYSpeed = calculateMeanSpeeds(states, obj, rawXSpeeds, rawYSpeeds)
                obj.xAccel = obj.meanXSpeed - prevObj.meanXSpeed
                obj.yAccel = obj.meanYSpeed - prevObj.meanYSpeed

        # calculate speed for offscreen objects for when they come
        # onscreen but don't add them to the observation
        if obj.relXPos > MAX_REL_X_POS or obj.yPos > MAX_Y_POS:
            continue

        xDistance = obj.xPos - curState.xPos
        yDistance = obj.yPos - curState.yPos
        euclideanDistance = np.linalg.norm(marioPos - np.array((obj.xPos, obj.yPos)))
        entities.append(
            (
                obj.typeID,
                np.array(
                    [
                        scaledEncoding(obj.relXPos, MAX_REL_X_POS, True),
                        scaledEncoding(obj.yPos, MAX_Y_POS, True),
                        scaledEncoding(xDistance, MAX_X_DISTANCE, False),
                        scaledEncoding(yDistance, MAX_Y_DISTANCE, False),
                        scaledEncoding(euclideanDistance, MAX_EUCLIDEAN_DISTANCE, True),
                        scaledEncoding(obj.meanXSpeed, ENTITY_MAX_MEAN_X_SPEED, False),
                        scaledEncoding(obj.meanYSpeed, ENTITY_MAX_MEAN_Y_SPEED, False),
                        scaledEncoding(obj.xAccel, ENTITY_MAX_MEAN_X_SPEED, False),
                        scaledEncoding(obj.yAccel, ENTITY_MAX_MEAN_Y_SPEED, False),
                        scaledEncoding(math.atan2(obj.meanXSpeed, obj.meanYSpeed), math.pi, False),
                    ],
                    dtype=np.float32,
                ),
            )
        )

    # sort entities by euclidean distance to mario
    sortedEntities = sorted(entities, key=lambda o: o[1][2])

    ids = np.asarray([i[0] for i in sortedEntities], dtype=np.uint8)
    paddingIDs = np.zeros((N_ENTITIES - len(ids)), dtype=np.uint8)
    allIDs = np.concatenate((ids, paddingIDs))

    entities = [i[1] for i in sortedEntities]
    paddingEntities = np.zeros((N_ENTITIES - len(entities), ENTITY_INFO_SIZE), dtype=np.float32)
    if len(entities) == 0:
        allEntities = paddingEntities
    else:
        allEntities = np.concatenate((entities, paddingEntities))

    return (marioInfo, allIDs, allEntities)


def findObjectInPrevState(obj: MarioLandObject, prevState: MarioLandGameState) -> MarioLandObject | None:
    prevObj: MarioLandObject = None
    prevObjs = [
        po
        for po in prevState.objects
        if obj.typeID == po.typeID
        and abs(obj.xPos - po.xPos) <= ENTITY_MAX_RAW_X_SPEED
        and abs(obj.yPos - po.yPos) <= ENTITY_MAX_RAW_Y_SPEED
    ]
    if len(prevObjs) == 1:
        prevObj = prevObjs[0]
    if len(prevObjs) > 1:
        prevObj = min(prevObjs, key=lambda po: abs(obj.xPos - po.xPos) + abs(obj.yPos - po.yPos))

    return prevObj


def calculateMeanSpeeds(
    states: Deque[MarioLandGameState], obj: MarioLandObject, rawXSpeeds: List[int], rawYSpeeds: List[int]
) -> Tuple[float, float]:
    # we already have the previous and current raw speeds
    for i in range(N_STATE_STACK - 2):
        state = states[i]
        xSpeed = 0
        ySpeed = 0
        prevObj = findObjectInPrevState(obj, state)
        if prevObj is not None:
            xSpeed = prevObj.rawXSpeed
            ySpeed = prevObj.rawYSpeed
        rawXSpeeds.append(xSpeed)
        rawYSpeeds.append(ySpeed)

    return np.mean(rawXSpeeds), np.mean(rawYSpeeds)


def getScalarFeatures(curState: MarioLandGameState, onGround: bool) -> np.ndarray:
    return np.array(
        np.concatenate(
            (
                oneHotEncoding(curState.powerupStatus, POWERUP_STATUSES),
                np.array(
                    [
                        float(onGround),
                        float(curState.hasStar),
                        scaledEncoding(curState.invincibleTimer, MAX_INVINCIBILITY_TIME, True),
                        scaledEncoding(curState.livesLeft, 99, True),
                        scaledEncoding(curState.coins, 99, True),
                        scaledEncoding(curState.timeLeft, 400, True),
                    ],
                    dtype=np.float32,
                ),
            ),
        )
    )


def scaledEncoding(val: int, max: int, minIsZero: bool) -> float:
    # if val > max:
    #     print(f"{val} > {max}")
    # elif minIsZero and val < 0:
    #     print(f"{val} < 0")
    # elif not minIsZero and val < -max:
    #     print(f"{val} < {-max}")

    scaled = 0.0
    if minIsZero:
        scaled = val / max
    else:
        # minimum value is less than zero, ensure scaling minimum is zero
        scaled = (val + max) / (max * 2)

    return np.clip(scaled, 0.0, 1.0)


def oneHotEncoding(val: int, max: int) -> np.ndarray:
    return np.squeeze(np.identity(max, dtype=np.float32)[val : val + 1])
