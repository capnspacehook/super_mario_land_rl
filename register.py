from pathlib import Path

import gymnasium as gym
from pufferlib.emulation import GymnasiumPufferEnv
from pyboy import PyBoy

from super_mario_land.super_mario_land import MarioLandEnv
from wrappers import MonitorTraining


def createSMLEnv(
    config,
    rom: Path = Path("games", "super_mario_land.gb"),
    render: bool = False,
    speed: int = 0,
    isEval: bool = False,
    isPlaytest: bool = False,
    isInteractiveEval: bool = False,
    cellID: int = 0,
) -> gym.Env:
    debug = False
    logLvl = "ERROR"
    if isPlaytest:
        # debug = True
        logLvl = "INFO"

    shouldRender = render or isPlaytest or isInteractiveEval
    if speed == 0 and isPlaytest:
        speed = 1

    pyboy = PyBoy(
        str(rom),
        window="SDL2" if shouldRender else "null",
        scale=4,
        debug=debug,
        log_level=logLvl,
    )
    pyboy.set_emulation_speed(speed)

    env = MarioLandEnv(
        pyboy,
        config,
        render=render,
        isEval=isEval,
        isPlaytest=isPlaytest,
        isInteractiveEval=isInteractiveEval,
        cellID=cellID,
    )

    if not isEval:
        env = MonitorTraining(env)

    return GymnasiumPufferEnv(env)
