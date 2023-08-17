from typing import List
import numpy as np

from bandits_framework import BanditsAlgorithm


class BanditStatistics:
    times_played: int = 0
    avg_reward: float = 0

    def update(self, reward: float):
        self.avg_reward = (self.times_played * self.avg_reward + reward) / (self.times_played + 1)
        self.times_played += 1


class UCB1BanditsAlgorithm(BanditsAlgorithm):
    def __init__(self, bandits_num: int):
        super().__init__(bandits_num)
        self._rounds_num = 0
        self._bandits_statistics = [BanditStatistics() for _ in range(bandits_num)]
        self._last_action = None

    def update(self, reward: float):
        self._bandits_statistics[self._last_action].update(reward)

    def choose_action(self) -> int:
        self._rounds_num += 1

        if self._rounds_num <= self._bandits_num:
            action = self._rounds_num - 1
        else:
            upper_confidence_bounds = self._calculate_upper_confidence_bounds()
            action = np.argmax(upper_confidence_bounds)

        self._last_action = action

        return action

    def _calculate_upper_confidence_bounds(self) -> List[float]:
        return [self._calculate_upper_confidence_bound(stats) for stats in self._bandits_statistics]

    def _calculate_upper_confidence_bound(self, stats: BanditStatistics) -> float:
        return stats.avg_reward + np.sqrt(2 * np.log(self._rounds_num) / stats.times_played)


class Exp3BanditAlgorithm(BanditsAlgorithm):
    def update(self, reward: float):
        pass

    def choose_action(self) -> int:
        pass
