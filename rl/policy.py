import numpy as np
from envs.discrete_MDP import DiscreteMDP


class Policy:
    def get_actions_distribution(self, state):
        raise NotImplementedError

    def sample_action(self, state):
        raise NotImplementedError


class DiscretePolicy(Policy):
    def __init__(self, states_num, actions_num):
        self.p = np.zeros([states_num, actions_num])

    def get_actions_distribution(self, state):
        return self.p[state, :]

    def sample_action(self, state):
        return np.random.choice(self.p.shape[1], p=self.get_actions_distribution(state))

    @classmethod
    def from_env(cls, env: DiscreteMDP):
        return cls(env.states_num, env.actions_num)


def make_greedy_policy(Q) -> DiscretePolicy:
    """
    Return greedy policy for Q function
    """
    policy = DiscretePolicy(Q.shape[0], Q.shape[1])

    for s in range(policy.p.shape[0]):
        policy.p[s, np.argmax(Q[s])] = 1

    return policy


def make_epsilon_greedy_policy(Q, eps: float) -> DiscretePolicy:
    """
    Return epsilon-greedy policy for Q function
    """
    policy = DiscretePolicy(Q.shape[0], Q.shape[1])
    greedy_policy = make_greedy_policy(Q)

    for s in range(policy.p.shape[0]):
        policy.p[s] = (1 - eps) * greedy_policy.p[s] + eps / policy.p.shape[1]

    return policy
