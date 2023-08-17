from typing import List, Tuple

import gymnasium as gym
from gymnasium.core import ObsType

from rl.policy import Policy


def sample_episode(env: gym.Env, policy: Policy):
    """
    Sample episode from environment with given policy

    Return list of (state, action, reward) tuples.
    """
    episode = []
    terminated = False
    truncated = False
    state, _ = env.reset()

    while not terminated and not truncated:
        action = policy.sample_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, float(reward)))
        state = next_state

    return episode


class EpisodeSampler:
    def __init__(self, env: gym.Env):
        self.env = env
        self.state = None
        self.done = True

    def reset(self):
        self.state, _ = self.env.reset()
        self.done = False

    def sample_episode(self, policy: Policy):
        episode = []
        self.reset()

        while True:
            state, action, reward, done = self.sample_one_step(policy)
            episode.append((state, action, reward))
            if done:
                break

        return episode

    def sample_one_step(self, policy: Policy):
        if not self.done:
            action = policy.sample_action(self.state)
            state = self.state
            self.state, reward, terminated, truncated, _ = self.env.step(action)
            reward = float(reward)
            if terminated or truncated:
                self.done = True
        else:
            state = None
            action = None
            reward = None

        return state, action, reward, self.done
