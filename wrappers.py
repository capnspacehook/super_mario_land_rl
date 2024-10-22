from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np


class VecRunningMean:
    def __init__(
        self,
        vecenv,
        epsilon: float = 1e-8,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
    ):
        self.vecenv = vecenv
        self._batch_size = self.vecenv.agents_per_batch

        self.rms = RunningMeanStd()
        self.clip_reward = clip_reward
        self.returns = np.zeros(self._batch_size)
        self.gamma = gamma
        self.epsilon = epsilon

    def __getattr__(self, name: str) -> Any:
        return getattr(self.vecenv, name)

    def recv(self):
        obs, rew, term, trunc, info, env_id, mask = self.vecenv.recv()

        self.returns = self.returns * self.gamma + rew
        self.rms.update(self.returns)

        rew = np.clip(rew / np.sqrt(self.rms.var + self.epsilon), -self.clip_reward, self.clip_reward)

        return obs, rew, term, trunc, info, env_id, mask


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        """
        self.mean = np.zeros((), np.float64)
        self.var = np.ones((), np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


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
            info["reward"] = self._reward
            info["length"] = self._steps
            self._steps = 0
            self._reward = 0.0

        return obs, rew, term, trunc, info
