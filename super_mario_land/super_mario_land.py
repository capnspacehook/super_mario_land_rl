from collections import deque
import hashlib
from math import floor
from io import BytesIO
from typing import Any, Deque, Dict, List, Tuple, Tuple
from os import listdir, stat
from os.path import isfile, join
import random
from pathlib import Path

import numpy as np
from gymnasium import Env
from gymnasium import spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from sqlalchemy import create_engine

from super_mario_land.constants import *
from super_mario_land.game_area import bouncing_boulder_tiles, worldTilesets
from super_mario_land.observation import getObservation
from super_mario_land.ram import *
from super_mario_land.settings import *
from database.state_manager import StateManager
from recorder import Recorder


worldToNextLevelState = {
    (1, 1): 2,
    (1, 2): 4,
    (1, 3): 6,
    (2, 1): 8,
    (2, 2): 10,
    (3, 1): 12,
    (3, 2): 14,
    (3, 3): 16,
    (4, 1): 18,
    (4, 2): 0,
}


class MarioLandEnv(Env):
    def __init__(
        self,
        pyboy: PyBoy,
        config,
        render: bool = False,
        isEval: bool = False,
        isPlaytest: bool = False,
        isInteractiveEval: bool = False,
        stateDir: Path = Path("states"),
    ) -> None:
        self.pyboy = pyboy
        self.prevState: MarioLandGameState | None = None
        self.isEval = isEval
        self.isPlaytest = isPlaytest
        self.isInteractiveEval = isInteractiveEval
        self.shouldRender = render or self.isEval or self.isPlaytest or self.isInteractiveEval
        self.shouldRecord = self.isEval and not self.isInteractiveEval
        self.interactive = False

        if self.isInteractiveEval:
            self.pyboy.set_emulation_speed(1)

        self.isEval = isEval
        self.maxLevel = MAX_START_LEVEL
        self.cellScore = 0.0
        self.cellID = 0
        self.cellCheckCounter = 0
        self.levelStr = ""
        self.invalidLevel = False
        self.evalStuck = 0
        self.evalNoProgress = 0
        self.invincibilityTimer = 0
        self.underground = False
        self.heartGetXPos = None
        self.heartFarming = False

        self.levelClearCounter = 0
        self.deathCounter = 0
        self.heartCounter = 0
        self.powerupCounter = 0
        self.coinCounter = 0

        self.episodeProgress = 0
        self.onGroundFor = 0

        self.rewardScale = config.reward_scale
        self.forwardRewardCoef = config.forward_reward
        self.progressRewardCoef = config.progress_reward
        self.backwardsPunishmentCoef = config.backwards_punishment
        self.waitReward = 0.25
        self.powerupReward = config.powerup_reward
        self.hitPunishment = config.hit_punishment
        self.heartReward = config.heart_reward
        self.movingPlatformXRewardCoef = 0.5
        self.movingPlatformYRewardCoef = 2
        self.levelClearReward = config.clear_level_reward
        self.levelClearTopReward = self.levelClearReward / 5
        self.levelClearLivesRewardCoef = self.levelClearReward / 10
        self.levelClearPowerupReward = self.levelClearReward / 20
        self.deathPunishment = config.death_punishment
        self.gameOverPunishment = self.deathPunishment * 1.33
        self.starPunishment = self.gameOverPunishment
        self.heartFarmingPunishment = self.gameOverPunishment
        self.coinReward = config.coin_reward
        self.scoreRewardCoef = config.score_reward
        self.clockPunishment = config.clock_punishment
        self.boulderReward = config.boulder_reward
        self.hitBossReward = 1.5
        self.killBossReward = 2.0

        self.gameStateCache: Deque[MarioLandGameState] = deque(maxlen=N_STATE_STACK)

        self.stateFiles = sorted([join(stateDir, f) for f in listdir(stateDir) if isfile(join(stateDir, f))])

        self._initLevelOrder()

        self._initDB()

        # setup action and observation spaces, initialize buttons
        self._initSpaces()
        self._initButtons()

        # compatibility with Env
        self.metadata["render_modes"] = ["rgb_array"]
        self.render_mode = None
        if render:
            self.render_mode = "rgb_array"

        self.recorder = None
        if self.isEval:
            self.recorder = Recorder(self.actions)

    def _initSpaces(self):
        self.actions = [
            [WindowEvent.PASS],
            [WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B],
            [
                WindowEvent.PRESS_ARROW_LEFT,
                WindowEvent.PRESS_BUTTON_B,
                WindowEvent.PRESS_BUTTON_A,
            ],
            [
                WindowEvent.PRESS_ARROW_RIGHT,
                WindowEvent.PRESS_BUTTON_B,
                WindowEvent.PRESS_BUTTON_A,
            ],
        ]

        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Dict(
            {
                GAME_AREA_OBS: spaces.Box(
                    low=0,
                    high=MAX_TILE,
                    shape=(GAME_AREA_HEIGHT, GAME_AREA_WIDTH),
                    dtype=np.uint8,
                ),
                MARIO_INFO_OBS: spaces.Box(low=0, high=1, shape=(MARIO_INFO_SIZE,), dtype=np.float32),
                ENTITY_ID_OBS: spaces.Box(low=0, high=MAX_ENTITY_ID, shape=(N_ENTITIES,), dtype=np.uint8),
                ENTITY_INFO_OBS: spaces.Box(
                    low=0, high=1, shape=(N_ENTITIES, ENTITY_INFO_SIZE), dtype=np.float32
                ),
                SCALAR_OBS: spaces.Box(low=0, high=1, shape=(SCALAR_SIZE,), dtype=np.float32),
            }
        )

    def _initButtons(self):
        # build list of possible inputs
        self._buttons = [
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_SELECT,
            WindowEvent.PRESS_BUTTON_START,
        ]
        self._held_buttons = {button: False for button in self._buttons}

        self._buttons_release = [
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_SELECT,
            WindowEvent.RELEASE_BUTTON_START,
        ]
        self._release_button = {
            button: r_button for button, r_button in zip(self._buttons, self._buttons_release)
        }

    def _initLevelOrder(self):
        # arrange the levels to be trained on in order of what level
        # is to be trained on first
        self.levelOrder = ["1-1", "1-2", "1-3", "2-1", "2-2", "3-1", "3-2", "3-3", "4-1", "4-2"]
        maxLevel = self.levelOrder.index(MAX_START_LEVEL)
        for _ in range(maxLevel):
            self.levelOrder.append(self.levelOrder.pop(0))

    def _initDB(self):
        engine = create_engine("postgresql+psycopg://postgres:password@localhost/postgres")
        self.stateManager = StateManager(engine)

        # only setup DB in the eval env so it's done once
        if not self.isEval or self.isInteractiveEval:
            return

        # create tables and indexes if they don't exist already
        self.stateManager.init_schema()
        # delete existing cells and cell scores from a previous run
        self.stateManager.delete_cells_and_cell_scores()

        for idx, name in enumerate(self.levelOrder):
            self.stateManager.insert_section(name, idx)

        # create cells from beginning of every level
        for i in range(0, 20, 2):
            stateFile = self.stateFiles[i]
            with open(stateFile, "rb") as f:
                self.pyboy.load_state(f)
                curState = self.gameState()
                cellHash, hashInput = self.cellHash(curState, isInitial=True)
                if self.stateManager.cell_exists(cellHash):
                    return

                levelName = f"{curState.world[0]}-{curState.world[1]}"
                levelIdx = self.levelOrder.index(levelName)
                f.seek(0)
                state = memoryview(f.read())

                self.stateManager.insert_initial_cell(
                    cellHash, hashInput, RANDOM_NOOP_FRAMES, levelIdx, state
                )

        # set starting max level
        maxLevelIdx = self.levelOrder.index(self.maxLevel)
        self.stateManager.upsert_max_section(maxLevelIdx)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)

        if self.prevState is None:
            self.prevState = self.gameState()

        self.prevState, prevAction = self._reset(options=options)

        self._held_buttons = {button: False for button in self._buttons}
        if prevAction is not None:
            actions = self.actions[prevAction]
            for button in actions:
                if button == WindowEvent.PASS:
                    continue
                release = self._release_button[button]
                self.pyboy.send_input(release)

        # not calling self.getObservation is fine here as self._reset will
        # fill self.observationCaches with the current observation
        return self.getObservation(), {}

    def _reset(self, options: dict[str, Any]) -> Tuple[MarioLandGameState, int]:
        # delete old cell score entries so querying the DB doesn't
        # slow too much
        if self.isEval:
            self.stateManager.delete_old_cell_scores()

        # reset counters
        self.cellScore = 0.0
        self.cellCheckCounter = 0
        self.episodeProgress = 0
        self.levelClearCounter = 0
        self.deathCounter = 0
        self.heartCounter = 0
        self.powerupCounter = 0
        self.coinCounter = 0

        # reset invalid level flag
        self.invalidLevel = False

        # load new cell
        if self.isEval:
            levelIdx = self.levelOrder.index(START_LEVEL)
            self.cellID, prevAction, maxNOOPs, initial, state = self.stateManager.get_first_cell(levelIdx)
        else:
            self.cellID, prevAction, maxNOOPs, initial, state = self.stateManager.get_random_cell()

        curState = self._loadLevel(state, maxNOOPs, initial=initial)

        # prevent extremely rare circumstances where an unsupported level
        # is loaded while during the no-op frames
        if self.levelStr not in LEVEL_END_X_POS:
            self.invalidLevel = True

        timer = STARTING_TIME
        if not self.isEval and not self.invalidLevel:
            # set the timer to a random time to make the environment more
            # stochastic; set it to lower values depending on where mario
            # is in the level is it's completable
            minTime = STARTING_TIME - int((curState.xPos / LEVEL_END_X_POS[self.levelStr]) * STARTING_TIME)
            minTime = max(MIN_RANDOM_TIME, minTime)
            timer = random.randint(minTime, STARTING_TIME)
            self._setTimer(timer)

        self._resetProgressAndCache(curState, True)

        return curState, prevAction

    def _loadLevel(
        self,
        state: memoryview,
        maxNOOPs: int,
        initial: bool = False,
        prevState: MarioLandGameState | None = None,
        transferState: bool = False,
    ) -> MarioLandGameState:
        with BytesIO(state) as bs:
            self.pyboy.load_state(bs)

        livesLeft = DEFAULT_LIVES_LEFT
        coins = DEFAULT_COINS
        score = 0

        if transferState:
            livesLeft = prevState.livesLeft
            coins = prevState.coins
            score = prevState.score
            if prevState.powerupStatus != STATUS_SMALL:
                self.pyboy.memory[POWERUP_STATUS_MEM_VAL] = 1
            if prevState.powerupStatus == STATUS_FIRE:
                self.pyboy.memory[HAS_FIRE_FLOWER_MEM_VAL] = 1

            curState = self.gameState()
            self._handlePowerup(curState)
        elif not self.isEval:
            # make starting lives random so NN can learn to strategically
            # handle lives
            livesLeft = random.randint(STARTING_LIVES_MIN, STARTING_LIVES_MAX) - 1

            # make starting coins random so NN can learn that collecting
            # 100 coins means a level up
            coins = random.randint(0, 99)

            # occasionally randomly set mario's powerup status so the NN
            # can learn to use the powerups; also makes the environment more
            # stochastic
            curState = self.gameState()
            if initial and random.randint(0, 100) < RANDOM_POWERUP_CHANCE:
                # 0: small with star
                # 1: big
                # 2: big with star
                # 3: fire flower
                # 4: fire flower with star
                gotStar = False
                randPowerup = random.randint(0, 4)
                # TODO: change back when pyboy bug is fixed
                if False:  # randPowerup in (0, 2, 4):
                    gotStar = True
                    self.pyboy.memory[STAR_TIMER_MEM_VAL] = 0xF8
                    # set star song so timer functions correctly
                    self.pyboy.memory[0xDFE8] = 0x0C
                if randPowerup != STATUS_SMALL:
                    self.pyboy.memory[POWERUP_STATUS_MEM_VAL] = 1
                    if randPowerup > 2:
                        self.pyboy.memory[HAS_FIRE_FLOWER_MEM_VAL] = 1

                curState = self.gameState()
                if gotStar:
                    curState.gotStar = True
                    curState.hasStar = True
                    curState.isInvincible = True
                self._handlePowerup(curState)
        else:
            if DEFAULT_POWERUP != STATUS_SMALL:
                self.pyboy.memory[POWERUP_STATUS_MEM_VAL] = 1
            if DEFAULT_POWERUP == STATUS_FIRE:
                self.pyboy.memory[HAS_FIRE_FLOWER_MEM_VAL] = 1
            curState = self.gameState()

        # set lives left
        livesTens = livesLeft // 10
        livesOnes = livesLeft % 10
        self.pyboy.memory[LIVES_LEFT_MEM_VAL] = (livesTens << 4) | livesOnes
        self.pyboy.memory[LIVES_LEFT_DISPLAY_MEM_VAL] = livesTens
        self.pyboy.memory[LIVES_LEFT_DISPLAY_MEM_VAL + 1] = livesOnes
        curState.livesLeft = livesLeft

        # set coins
        self.pyboy.memory[COINS_MEM_VAL] = dec_to_bcm(coins)
        self.pyboy.memory[COINS_DISPLAY_MEM_VAL] = coins // 10
        self.pyboy.memory[COINS_DISPLAY_MEM_VAL + 1] = coins % 10
        curState.coins = coins

        # set score
        if not self.isEval or score == 0:
            # set score to 0
            for i in range(3):
                self.pyboy.memory[SCORE_MEM_VAL + i] = 0
            for i in range(5):
                self.pyboy.memory[SCORE_DISPLAY_MEM_VAL + i] = 44
            self.pyboy.memory[SCORE_DISPLAY_MEM_VAL + 5] = 0
        else:
            scoreTens = score % 100
            scoreHundreds = (score // 100) % 100
            scoreTenThousands = score // 10000
            self.pyboy.memory[SCORE_MEM_VAL] = dec_to_bcm(scoreTens)
            self.pyboy.memory[SCORE_MEM_VAL + 1] = dec_to_bcm(scoreHundreds)
            self.pyboy.memory[SCORE_MEM_VAL + 2] = dec_to_bcm(scoreTenThousands)

            paddedScore = f"{score:06}"
            leadingZerosReplaced = True
            for i in range(6):
                digit = paddedScore[i]
                if leadingZerosReplaced and digit == "0":
                    self.pyboy.memory[SCORE_DISPLAY_MEM_VAL + i] = 44
                else:
                    leadingZerosReplaced = False
                    self.pyboy.memory[SCORE_DISPLAY_MEM_VAL + i] = int(digit)

        if not self.isEval and len(curState.objects) != 0:
            # do nothing for a random amount of frames to make entity
            # placements varied and the environment more stochastic
            nopFrames = random.randint(0, maxNOOPs)
            if nopFrames != 0:
                # release any buttons in case any where being held before
                self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
                self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)

                self.pyboy.tick(count=nopFrames, render=False)
                curState = self.gameState()
                # set cell as invalid just in case a cell was added that
                # will cause mario to die almost immediately
                if not curState.onGround or self._isDead(curState):
                    self.stateManager.set_cell_invalid(self.cellID)

        # reset max level progress
        curState.levelProgressMax = curState.xPos

        # reset death counter
        if not self.isEval:
            self.deathCounter = 0

        # reset heart farming vars
        self.heartGetXPos = None
        self.heartFarming = False

        # set game area mapping
        self.levelStr = f"{curState.world[0]}-{curState.world[1]}"
        self.pyboy.game_area_mapping(worldTilesets[curState.world[0]])

        return curState

    def _setTimer(self, timer: int):
        timerHundreds = timer // 100
        timerTens = timer - (timerHundreds * 100)
        self.pyboy.memory[TIMER_HUNDREDS] = timerHundreds
        self.pyboy.memory[TIMER_TENS] = dec_to_bcm(timerTens)
        self.pyboy.memory[TIMER_FRAMES] = 0x28

    def _resetProgressAndCache(self, curState: MarioLandGameState, resetCaches: bool):
        self.evalStuck = 0
        self.evalNoProgress = 0
        self.underground = False
        self.onGroundFor = 0

        # reset level progress since death
        curState.levelProgressSinceDeath = curState.xPos

        # reset the game state cache
        if resetCaches:
            [self.gameStateCache.append(curState) for _ in range(N_STATE_STACK)]

        if not resetCaches:
            curState.posReset = True
            self.gameStateCache.append(curState)

    def getObservation(self) -> Any:
        return getObservation(self.pyboy, self.gameStateCache, self.onGroundFor == ON_GROUND_FRAMES)

    def step(self, actionIdx: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        # send actions
        actions = self.actions[actionIdx]
        if self.isInteractiveEval and Path("agent_enabled.txt").is_file():
            with open("agent_enabled.txt", "r") as f:
                if "t" in f.read():
                    self.interactive = False
                    self.sendInputs(actions)
                elif not self.interactive:
                    self.sendInputs([WindowEvent.PASS])
                    self.interactive = True
        elif not self.isPlaytest:
            self.sendInputs(actions)

        # handle frame skip here to avoid computing unused observations
        totalReward = 0.0
        for i in range(FRAME_SKIP):
            # advance frame
            pyboyStillRunning = self.pyboy.tick(render=self.shouldRender)

            # update game state cache and on ground info
            curState = self.gameState(self.prevState)
            self.gameStateCache.append(curState)

            if not curState.onGround:
                self.onGroundFor = 0
            elif self.onGroundFor < ON_GROUND_FRAMES:
                self.onGroundFor += 1

            # compute reward, handle dying or level completion
            reward = self.reward(actions, curState) * self.rewardScale
            totalReward += reward

            if self.shouldRecord:
                self.recorder.recordStep(self.render, actionIdx, reward)

            curState = self.handleProgression(curState)

            terminated = not pyboyStillRunning or self.terminated(curState)
            truncated = self.truncated(curState)
            done = terminated or truncated
            # save current state as a cell, record cell score if the episode
            # has ended
            if not self.isPlaytest:
                self.handleCells(curState, int(actionIdx), reward, done)

            # don't update the previous game state yet if we need to
            # compute an observation so it can use the actual previous
            # game state
            if done:
                if self.shouldRecord:
                    self.recorder.episodeDone()
                break
            elif i != FRAME_SKIP - 1:
                self.prevState = curState

        obs = self.getObservation()
        info = {}
        if done:
            info = self.info(curState)

        if self.isPlaytest:
            self.printGameState(obs, totalReward / self.rewardScale, curState)

        self.prevState = curState

        return obs, totalReward, terminated, truncated, info

    def reward(self, actions: List[int], curState: MarioLandGameState) -> float:
        # return punishment on mario's death
        if self._isDead(curState):
            self.deathCounter += 1

            if curState.livesLeft == 0:
                return self.gameOverPunishment

            return self.deathPunishment

        # handle level clear
        if curState.statusTimer == TIMER_LEVEL_CLEAR or curState.gameState == 5:
            levelClear = self.levelClearReward
            # reward clearing a level through the top spot
            if curState.yPos > 60:
                levelClear += self.levelClearTopReward
            # reward clearing a level with extra lives
            levelClear += curState.livesLeft * self.levelClearLivesRewardCoef
            # reward clearing a level while powered up
            if curState.powerupStatus == STATUS_BIG:
                levelClear += self.levelClearPowerupReward
            elif curState.powerupStatus == STATUS_FIRE:
                levelClear += self.levelClearPowerupReward

            return levelClear

        # the game registers mario as on the ground a couple of frames
        # before he actually is to change his pose
        onGround = self.onGroundFor == ON_GROUND_FRAMES

        # add time punishment every step to encourage speed more
        clock = self.clockPunishment

        # reward level progress
        progress = np.clip(
            curState.levelProgressSinceDeath - self.prevState.levelProgressSinceDeath, 0, MARIO_MAX_X_SPEED
        )
        progress = progress * self.progressRewardCoef

        # reward or punish depending on the speed and direction mario
        # is traveling
        xSpeed = np.clip(curState.xPos - self.prevState.xPos, -MARIO_MAX_X_SPEED, MARIO_MAX_X_SPEED)
        movement = 0
        if xSpeed > 0:
            movement = xSpeed * self.forwardRewardCoef
        elif xSpeed < 0:
            movement = -xSpeed * self.backwardsPunishmentCoef

        wait = 0
        if onGround and actions == [WindowEvent.PASS]:
            wait = self.waitReward

        # reward coins increasing
        if curState.coins >= self.prevState.coins:
            collectedCoins = curState.coins - self.prevState.coins
        else:
            collectedCoins = (99 - self.prevState.coins) + curState.coins
        coins = (collectedCoins) * self.coinReward
        self.coinCounter += collectedCoins

        # reward standing on moving platforms if the platforms are moving
        # forward or upwards to encourage waiting on platforms until a
        # more optimal jump can be made
        movingPlatform = 0
        _, onMovingPlatform = self._standingOnMovingPlatform(curState)
        # don't reward when moving forward on a moving platform
        if onGround and onMovingPlatform and actions == [WindowEvent.PASS]:
            ySpeed = np.clip(curState.yPos - self.prevState.yPos, -MARIO_MAX_Y_SPEED, MARIO_MAX_Y_SPEED)
            movingPlatform += max(0, xSpeed) * self.movingPlatformXRewardCoef
            movingPlatform += max(0, ySpeed) * self.movingPlatformYRewardCoef

        # in world 3 reward standing on bouncing boulders to encourage
        # waiting for them to fall and ride on them instead of immediately
        # jumping into spikes, but only if the boulders are moving to the
        # right
        standingOnBoulder = 0
        if (
            curState.world[0] == 3
            and onGround
            and xSpeed > 0
            and actions == [WindowEvent.PASS]
            and self._standingOnTiles(bouncing_boulder_tiles)
        ):
            standingOnBoulder = self.boulderReward

        # reward getting powerups and manage powerup related bookkeeping
        powerup = self._handlePowerup(curState)
        if powerup > 0:
            self.powerupCounter += 1

        heart = 0
        if curState.livesLeft > self.prevState.livesLeft:
            # discourage heart farming, punish getting a heart in the
            # same place twice
            if (
                self.heartGetXPos is not None
                and curState.xPos + HEART_FARM_X_POS_MULTIPLE >= self.heartGetXPos
                and curState.xPos - HEART_FARM_X_POS_MULTIPLE <= self.heartGetXPos
            ):
                self.heartFarming = True
                heart = self.heartFarmingPunishment
            else:
                # reward getting 1-up
                heart = (curState.livesLeft - self.prevState.livesLeft) * self.heartReward
                self.heartCounter += curState.livesLeft - self.prevState.livesLeft

            self.heartGetXPos = curState.xPos

        # reward killing enemies (and breaking blocks I guess)
        score = 0
        if curState.score > self.prevState.score:
            # don't reward collecting coins or powerups twice
            score = curState.score - self.prevState.score
            score -= 100 * collectedCoins
            if powerup > 0:
                score -= 1000
            score *= self.scoreRewardCoef

        # reward damaging or killing a boss
        boss = 0
        if curState.bossActive and curState.bossHealth < self.prevState.bossHealth:
            if self.prevState.bossHealth - curState.bossHealth == 1:
                boss = self.hitBossReward
            elif curState.bossHealth == 0:
                boss = self.killBossReward

        reward = (
            clock
            + progress
            + movement
            + wait
            + coins
            + movingPlatform
            + standingOnBoulder
            + powerup
            + heart
            + score
            + boss
        )

        return reward

    def _standingOnMovingPlatform(self, curState: MarioLandGameState) -> Tuple[MarioLandObject | None, bool]:
        for obj in curState.objects:
            if obj.typeID == MOVING_PLATFORM_TYPE_ID and curState.yPos - 10 == obj.yPos:
                return obj, True
        return None, False

    def _standingOnTiles(self, tiles: List[int]) -> bool:
        sprites = self.pyboy.get_sprite_by_tile_identifier(tiles, on_screen=True)
        if len(sprites) == 0:
            return False

        leftMarioLeg = self.pyboy.get_sprite(5)
        leftMarioLegXPos = leftMarioLeg.x
        rightMarioLegXPos = leftMarioLegXPos + 8
        marioLegsYPos = leftMarioLeg.y
        if leftMarioLeg.attr_x_flip:
            rightMarioLegXPos = leftMarioLegXPos
            leftMarioLegXPos -= 8

        for spriteIdxs in sprites:
            for spriteIdx in spriteIdxs:
                sprite = self.pyboy.get_sprite(spriteIdx)
                # y positions are inverted for some reason
                if (marioLegsYPos + 6 <= sprite.y and marioLegsYPos + 10 >= sprite.y) and (
                    (leftMarioLegXPos >= sprite.x - 4 and leftMarioLegXPos <= sprite.x + 4)
                    or (rightMarioLegXPos >= sprite.x - 4 and rightMarioLegXPos <= sprite.x + 4)
                ):
                    return True

        return False

    def _handlePowerup(self, curState: MarioLandGameState) -> int:
        powerup = 0
        if curState.gotStar:
            self.invincibilityTimer = STAR_TIME
            # The actual star timer is set to 248 and only ticks down
            # when the frame counter is a one greater than a number
            # divisible by four. Don't ask me why. This accounts for
            # extra invincibility frames depending on what the frame
            # counter was at when the star was picked up
            frames = self.pyboy.memory[FRAME_COUNTER_MEM_VAL]
            extra = (frames - 1) % 4
            self.invincibilityTimer += extra
            powerup += self.starPunishment
        if curState.hasStar:
            # current powerup status will be set to star, so set it to
            # the powerup of the last frame so the base powerup is accurate
            curState.powerupStatus = self.prevState.powerupStatus

        # big reward for acquiring powerups, small punishment for
        # loosing them but not too big a punishment so abusing
        # invincibility frames isn't discouraged
        if curState.powerupStatus != self.prevState.powerupStatus:
            if self.prevState.powerupStatus == STATUS_SMALL:
                # mario got a mushroom
                powerup = self.powerupReward
            elif self.prevState.powerupStatus == STATUS_GROWING and curState.powerupStatus == STATUS_SMALL:
                # mario got hit while growing from a mushroom
                powerup = self.hitPunishment
            elif self.prevState.powerupStatus == STATUS_BIG:
                if curState.powerupStatus == STATUS_FIRE:
                    powerup = self.powerupReward
                elif curState.powerupStatus == STATUS_SMALL:
                    self.invincibilityTimer = SHRINK_TIME
                    powerup = self.hitPunishment
            elif self.prevState.powerupStatus == STATUS_FIRE:
                # mario got hit and lost the fire flower
                self.invincibilityTimer = SHRINK_TIME
                powerup = self.hitPunishment

        if self.invincibilityTimer != 0:
            curState.invincibleTimer = self.invincibilityTimer
            self.invincibilityTimer -= 1

        return powerup

    def handleProgression(self, curState: MarioLandGameState) -> MarioLandGameState:
        if self._isDead(curState):
            if curState.livesLeft == 0:
                # no lives left, just return so this episode can be terminated
                return curState

            timeLeft = curState.timeLeft

            # skip frames where mario is dying
            statusTimer = curState.statusTimer
            gameState = curState.gameState
            while gameState in (3, 4) or (gameState == 1 and statusTimer != 0):
                self.pyboy.tick(render=False)
                gameState = self.pyboy.memory[GAME_STATE_MEM_VAL]
                statusTimer = self.pyboy.memory[STATUS_TIMER_MEM_VAL]

            self.pyboy.tick(count=5, render=False)

            curState = self.gameState(curState)

            # don't let the game set the timer back to max time
            timer = np.clip(timeLeft - DEATH_TIME_LOSS, MIN_TIME, STARTING_TIME)
            self._setTimer(timer)

            # don't reset state and observation caches so the agent can
            # see that it died
            self._resetProgressAndCache(curState, False)

            return curState

        if curState.statusTimer == TIMER_LEVEL_CLEAR:
            self.levelClearCounter += 1

            # load the next level directly to avoid processing
            # unnecessary frames and the AI playing levels we
            # don't want it to
            stateFile = self.stateFiles[worldToNextLevelState[curState.world]]
            with open(stateFile, "rb") as f:
                state = memoryview(f.read())
                # keep lives and powerup in new level
                curState = self._loadLevel(
                    state, RANDOM_NOOP_FRAMES, prevState=self.prevState, transferState=True
                )

            # don't reset state and observation caches so the agent can
            # see that it started a new level
            self._setTimer(STARTING_TIME)
            self._resetProgressAndCache(curState, False)

            return curState

        if curState.pipeWarping:
            gameState = curState.gameState
            while gameState != 0:
                self.pyboy.tick(render=False)
                gameState = self.pyboy.memory[GAME_STATE_MEM_VAL]

            return self.gameState(curState)

        # keep track of how long the agent is idle so we can end early
        # in an evaluation
        if self.isEval:
            if curState.xPos == self.prevState.xPos:
                self.evalStuck += 1
            else:
                self.evalStuck = 0

            if curState.levelProgressSinceDeath == self.prevState.levelProgressSinceDeath:
                self.evalNoProgress += 1
            else:
                self.evalNoProgress = 0

        return curState

    def terminated(self, curState: MarioLandGameState) -> bool:
        return self._isDead(curState) and curState.livesLeft == 0

    def truncated(self, curState: MarioLandGameState) -> bool:
        # If Mario has not moved in 10s, end the eval episode.
        # If no forward progress has been made in 20s, end the eval episode.
        return (
            self.heartFarming
            or curState.hasStar  # TODO: remove once star bug has been fixed
            or (
                (self.isEval and not self.isInteractiveEval)
                and (self.evalStuck == 600 or self.evalNoProgress == 1200)
            )
            or self.invalidLevel
        )

    def _isDead(self, curState: MarioLandGameState) -> bool:
        return curState.gameState in GAME_STATES_DEAD

    def handleCells(self, curState: MarioLandGameState, action: int, reward: float, done: bool):
        if self.isEval:
            if done and self.levelStr > self.maxLevel:
                self.maxLevel = self.levelStr
                maxLevelIdx = self.levelOrder.index(self.maxLevel)
                self.stateManager.update_max_section(maxLevelIdx)
            return

        self.cellScore += reward

        if done:
            self.stateManager.record_score(self.cellID, float(self.cellScore))
            return

        # only check if this cell is new every N frames to avoid
        # making DB queries every frame
        if self.cellCheckCounter != FRAME_CELL_CHECK:
            self.cellCheckCounter += 1
            return
        self.cellCheckCounter = 0

        cellHash, hashInput = self.cellHash(curState)
        if cellHash is not None and not self.stateManager.cell_exists(cellHash):
            with BytesIO() as state:
                self.pyboy.save_state(state)
                state.seek(0)

                maxNOOPs = RANDOM_NOOP_FRAMES

                # ensure loading from the state won't instantly kill mario
                if any((obj.typeID in ENEMY_TYPE_IDS for obj in curState.objects)):
                    self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
                    self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)

                    unsafeState = False
                    # if mario dies without moving in less than 2 seconds
                    # don't save the state; we don't want to load states
                    # that will result in an unpreventable death
                    for _ in range(20):
                        self.pyboy.tick(count=6, render=False)
                        if (
                            self.pyboy.memory[GAME_STATE_MEM_VAL] != 0
                            or self.pyboy.memory[MARIO_ON_GROUND_MEM_VAL] == 0
                        ):
                            unsafeState = True
                            break

                        # don't save states where an enemy gets really close
                        # to mario, agents will likely almost always die when
                        # loading the state skewing training
                        for obj in curState.objects:
                            if obj.typeID in ENEMY_TYPE_IDS:
                                x = np.array((curState.xPos, obj.xPos))
                                y = np.array((curState.yPos, obj.yPos))
                                distance = np.linalg.norm(x - y)
                                if distance < EMEMY_SAFE_DISTANCE:
                                    unsafeState = True
                                    break

                    self.pyboy.load_state(state)
                    if unsafeState:
                        return

                    state.seek(0)
                    maxNOOPs = RANDOM_NOOP_FRAMES_WITH_ENEMIES

                try:
                    levelName = f"{curState.world[0]}-{curState.world[1]}"
                    levelIdx = self.levelOrder.index(levelName)
                    self.stateManager.insert_cell(
                        cellHash, hashInput, action, maxNOOPs, levelIdx, state.getbuffer()
                    )
                except Exception as e:
                    print(e)

    def cellHash(self, curState: MarioLandGameState, isInitial=False) -> Tuple[str | None, str | None]:
        # don't save states when mario isn't on the ground or if mario
        # isn't controllable
        if not isInitial and (
            self.onGroundFor != ON_GROUND_FRAMES
            or curState.gameState != 0
            or curState.xPos > LEVEL_END_X_POS[self.levelStr] - 30
        ):
            return None, None

        # don't save states when a star is present
        for obj in curState.objects:
            if obj.typeID == STAR_TYPE_ID:
                return None, None

        roundedXPos = X_POS_MULTIPLE * floor(curState.xPos / X_POS_MULTIPLE)
        # avoid tons of different cells from mario just riding a platform
        # that are basically the same
        roundedYPos = 0
        _, onMovingPlat = self._standingOnMovingPlatform(curState)
        if not onMovingPlat:
            roundedYPos = Y_POS_MULTIPLE * floor(curState.yPos / Y_POS_MULTIPLE)

        objectTypes = ""
        # sort objects by the type ID to prevent different cells that
        # are basically the same but have objects in a different order
        objects = sorted(curState.objects, key=lambda o: o.typeID)
        for obj in objects:
            objRoundedXPos = ENTITY_X_POS_MULTIPLE * floor(obj.xPos / ENTITY_X_POS_MULTIPLE)
            objRoundedYPos = 0
            if obj.typeID not in ENTITIES_IGNORE_Y_POS:
                objRoundedYPos = ENTITY_Y_POS_MULTIPLE * floor(obj.yPos / ENTITY_Y_POS_MULTIPLE)
            objectTypes += f"{obj.typeID}|{objRoundedXPos}|{objRoundedYPos}/"

        input = f"{curState.world}|{roundedXPos}|{roundedYPos}|{curState.powerupStatus}/{objectTypes}"
        return hashlib.md5(input.encode("utf-8")).hexdigest(), input

    def sendInputs(self, actions: list[int]):
        # release buttons that were pressed in the past that are not
        # still being pressed
        currentlyHeld = [b for b in self._held_buttons if self._held_buttons[b]]
        for heldButton in currentlyHeld:
            if heldButton not in actions:
                release = self._release_button[heldButton]
                self.pyboy.send_input(release)
                self._held_buttons[heldButton] = False

        # press buttons we want to press
        for button in actions:
            if button == WindowEvent.PASS:
                continue
            self.pyboy.send_input(button)
            # update status of the button
            self._held_buttons[button] = True

    def gameState(self, prevState: MarioLandGameState | None = None):
        curState = MarioLandGameState(self.pyboy)
        if prevState is None:
            return curState

        curState.levelProgressMax = max(self.prevState.levelProgressMax, curState.xPos)
        curState.levelProgressSinceDeath = max(self.prevState.levelProgressSinceDeath, curState.xPos)
        stepProgress = curState.levelProgressMax - self.prevState.levelProgressMax
        self.episodeProgress += stepProgress if stepProgress > 0 else 0

        return curState

    def info(self, curState: MarioLandGameState) -> Dict[str, Any]:
        return {
            "progress": self.episodeProgress,
            "levels_cleared": self.levelClearCounter,
            "deaths": self.deathCounter,
            "hearts": self.heartCounter,
            "powerups": self.powerupCounter,
            "coins": self.coinCounter,
            "score": curState.score,
        }

    def render(self):
        return self.pyboy.screen.image.copy()

    def printGameState(self, obs: Dict[str, Any], reward: float, curState: MarioLandGameState):
        objects = ""
        for i, o in enumerate(curState.objects):
            objects += f"{i}: {o.typeID} {o.xPos} {o.yPos} {round(o.meanXSpeed,3)} {round(o.meanYSpeed,3)} {round(o.xAccel,3)} {round(o.yAccel,3)}\n"

        s = f"""
Game area:
{obs[GAME_AREA_OBS]}

Reward: {reward}
Episode progress: {self.episodeProgress}
Max level progress: {curState.levelProgressMax}
Powerup: {curState.powerupStatus}
Lives left: {curState.livesLeft}
Score: {curState.score}
Status timer: {curState.statusTimer} {self.pyboy.memory[STAR_TIMER_MEM_VAL]} {self.pyboy.memory[0xDA00]}
X, Y: {curState.xPos}, {curState.yPos}
Rel X, Y {curState.relXPos} {curState.relYPos}
Speeds: {round(curState.meanXSpeed, 3)} {round(curState.meanYSpeed, 3)} {round(curState.xAccel, 3)} {round(curState.yAccel, 3)}
Invincibility: {curState.gotStar} {curState.hasStar} {curState.isInvincible} {curState.invincibleTimer}
Boss: {curState.bossActive} {curState.bossHealth}
Game state: {curState.gameState}
{objects}
"""
        print(s[1:], flush=True)

    def close(self):
        self.stateManager.close()
