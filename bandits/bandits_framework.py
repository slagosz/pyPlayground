import random
from typing import List


class Bandit:
    def play(self) -> float:
        raise NotImplementedError

    def get_params(self) -> dict:
        raise NotImplementedError


class BernoulliBandit(Bandit):
    def __init__(self, p: float):
        self._p = p

    def play(self) -> float:
        r = random.uniform(0, 1)
        if r < self._p:
            return 1.0
        else:
            return 0.0

    def get_params(self) -> dict:
        return dict(p=self._p)


class BanditsProblem:
    def __init__(self, bandits: List[Bandit]):
        self._bandits = bandits

    def play(self, bandit_index: int) -> float:
        r = self._bandits[bandit_index].play()

        return r

    def optimal_action_expected_reward(self) -> float:
        raise NotImplementedError

    def optimal_action(self) -> int:
        raise NotImplementedError


class BernoulliBanditsProblem(BanditsProblem):
    def __init__(self, bandits_num: int):
        bandits = [BernoulliBandit(random.uniform(0, 1)) for _ in range(bandits_num)]
        super().__init__(bandits)

        self._optimal_bandit = min(bandits, key=lambda b: b.get_params()['p'])
        self._optimal_bandit_index = bandits.index(self._optimal_bandit)

    def optimal_action_expected_reward(self) -> float:
        return self._optimal_bandit.get_params()['p']

    def optimal_action(self) -> int:
        return self._optimal_bandit_index


class BanditsAlgorithm:
    def update(self, r: float):
        raise NotImplementedError

    def choose_action(self) -> int:
        raise NotImplementedError


class BanditsExperiment:
    def __init__(self, problem: BanditsProblem, algorithm: BanditsAlgorithm):
        self._problem = problem
        self._algorithm = algorithm

    def run(self, rounds_num: int) -> float:
        total_reward = 0
        for t in range(rounds_num):
            action = self._algorithm.choose_action()
            r = self._problem.play(action)
            total_reward += r
            self._algorithm.update(r)

        # optimal_expected_reward = self._problem.optimal_action_expected_reward() * rounds_num
        # regret = optimal_expected_reward - total_reward

        return total_reward



