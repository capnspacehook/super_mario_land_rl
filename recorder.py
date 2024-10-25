from typing import Any, Callable, List

import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import ImageFont
from PIL import ImageDraw
import psutil
from pyboy.utils import WindowEvent


actionsToText = {
    WindowEvent.PASS: "NOTHING",
    WindowEvent.PRESS_ARROW_LEFT: "LEFT",
    WindowEvent.PRESS_ARROW_RIGHT: "RIGHT",
    WindowEvent.PRESS_ARROW_UP: "UP",
    WindowEvent.PRESS_ARROW_DOWN: "DOWN",
    WindowEvent.PRESS_BUTTON_B: "B",
    WindowEvent.PRESS_BUTTON_A: "A",
    WindowEvent.PRESS_BUTTON_START: "START",
    WindowEvent.PRESS_BUTTON_SELECT: "SELECT",
}


class Recorder:
    def __init__(
        self,
        actions: List[List[int]],
        episode_num: int = 1,
        native_fps: int = 60,
        rec_steps: int = 2,
        reward_steps: int = 4,
    ):
        self._cores = psutil.cpu_count(logical=True)

        self.actions = actions
        self.episode_num = episode_num
        self.native_fps = native_fps
        self.rec_steps = rec_steps
        self.reward_steps = reward_steps
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf", 12)

        self.frames = []
        self.count_episode = 0
        self.cur_step = 0
        self.cur_reward = 0
        self.last_reward = 0

    def recordStep(self, render: Callable[[], Any], action: int, reward: float):
        self.cur_reward += reward
        if self.cur_step % self.reward_steps == self.reward_steps - 1:
            self.last_reward = self.cur_reward
            self.cur_reward = 0
        if self.cur_step % self.rec_steps == 0:
            self._recordStep(render, action)
        self.cur_step += 1

    def _recordStep(self, render: Callable[[], Any], action: int):
        actionList = self.actions[action]
        actionText = ""
        for action in actionList:
            actionText += actionsToText[action] + ", "

        frame = render()
        draw = ImageDraw.Draw(frame)
        draw.text((0, 17), f"ACTION: {actionText[:-2]}", (0, 102, 255), self.font)
        draw.text((0, 30), f"REWARD: {round(self.last_reward, 6)}", (0, 102, 255), self.font)
        self.frames.append(np.array(frame))

    def episodeDone(self):
        self.count_episode += 1
        if self.count_episode == self.episode_num:
            self._stop_recording()

    def _stop_recording(self):
        v = ImageSequenceClip(self.frames, fps=self.native_fps / self.rec_steps)
        v.write_videofile(
            filename="/tmp/eval.mp4",
            codec="libx264",
            bitrate="4000k",
            preset="slow",
            threads=self._cores,
            logger=None,
        )
        v.close()

        self.frames = []
        self.count_episode = 0
        self.cur_step = 0
        self.cur_reward = 0
        self.last_reward = 0
