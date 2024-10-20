from typing import Any

import gymnasium as gym


class MonitorTraining(gym.Wrapper):
    def __init__(self, env: gym.Env):
        self._steps = 0
        self._reward = 0.0

        super().__init__(env)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, rew, term, trunc, info = self.env.step(action)

        self._steps += 1
        self._reward += rew

        if term or trunc:
            info["mean_reward"] = self._reward
            info["mean_length"] = self._steps
            self._steps = 0
            self._reward = 0.0

        return obs, rew, term, trunc, info
