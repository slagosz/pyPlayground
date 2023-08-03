import numpy as np
from envs.discrete_MDP import DiscreteMDP


class DiscretePolicy:
    def __init__(self, env: DiscreteMDP):
        self.p = np.ones([env.states_array.shape[0], env.action_space.n]) / env.action_space.n

    def get_actions_distribution(self, state):
        return self.p[state, :]

    def get_action(self, state):
        return np.random.choice(self.p[state], p=self.get_actions_distribution(state))


def policy_evaluation(policy: DiscretePolicy, env: DiscreteMDP, stop_threshold: float):
    state_to_value_array = np.zeros_like(env.states_array, dtype=float)

    while True:
        delta = 0

        for s, V in enumerate(state_to_value_array):
            prev_V = V
            V = (policy.get_actions_distribution(s) @ (env.p[:, s, :].T * (env.r[:, s, :].T + state_to_value_array))).sum()
            #   (1, a)                                (a, s)              (a, s)             (1, s)
            state_to_value_array[s] = V
            delta = max(delta, np.abs(prev_V - V))

        if delta < stop_threshold:
            break

    return state_to_value_array


def V2Q(v, env: DiscreteMDP):
    q = np.zeros()