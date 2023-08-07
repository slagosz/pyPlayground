import numpy as np
from envs.discrete_MDP import DiscreteMDP


class DiscretePolicy:
    def __init__(self, states_num, actions_num):
        self.p = np.zeros([states_num, actions_num])

    def get_actions_distribution(self, state):
        return self.p[state, :]

    def sample_action(self, state):
        return np.random.choice(self.p[state], p=self.get_actions_distribution(state))

    @classmethod
    def from_env(cls, env: DiscreteMDP):
        return cls(env.states_num, env.actions_num)


def policy_evaluation(policy: DiscretePolicy, env: DiscreteMDP, stop_threshold: float, discount_factor: float = 1.0):
    """
    Evaluate policy. Return V function: state -> V value
    """
    V_array = np.zeros_like(env.states_array, dtype=float)  # state -> V map

    while True:
        delta = 0

        for s, V in enumerate(V_array):
            prev_V = V
            V = (policy.get_actions_distribution(s) @ (env.p[:, s, :].T * (env.r[:, s, :].T + discount_factor * V_array))).sum()
            #   (1, a)                                (a, s)              (a, s)              (1, s)
            V_array[s] = V
            delta = max(delta, np.abs(prev_V - V))

        if delta < stop_threshold:
            break

    return V_array


def V2Q(V, env: DiscreteMDP, discount_factor: float = 1.0):
    """
    Return Q function: (state, action) -> Q value
    """
    Q = ((env.p.T * discount_factor * V).T + env.r * env.p).sum(0)

    return Q


def Q2V(Q, policy: DiscretePolicy):
    """
    Return V function for Q function and policy
    """
    V = (policy.p * Q).sum(1)

    return V


def make_greedy_policy(Q) -> DiscretePolicy:
    """
    Return greedy policy for Q function
    """
    policy = DiscretePolicy(Q.shape[0], Q.shape[1])

    for s in range(policy.p.shape[0]):
        policy.p[s, np.argmax(Q[s])] = 1

    return policy


def policy_iteration(env: DiscreteMDP, eval_stop_threshold: float, max_iterations: int, discount_factor: float) -> DiscretePolicy:
    """
    Policy iteration algorithm
    """
    policy = DiscretePolicy.from_env(env)

    for _ in range(max_iterations):
        V = policy_evaluation(policy, env, eval_stop_threshold, discount_factor=discount_factor)
        Q = V2Q(V, env)
        new_policy = make_greedy_policy(Q)

        if np.all(policy.p == new_policy.p):
            break

        policy = new_policy

    return policy


def value_iteration(env: DiscreteMDP, stop_threshold: float, discount_factor: float) -> DiscretePolicy:
    """
    Value iteration algorithm
    """
    V = np.zeros_like(env.states_array, dtype=float)

    while True:
        Q = V2Q(V, env, discount_factor=discount_factor)
        prev_V = V
        V = np.max(Q, axis=1)

        delta = max(np.abs(prev_V - V))

        if delta < stop_threshold:
            break

    policy = make_greedy_policy(Q)

    return policy


def value_iteration_finer_granularity(env: DiscreteMDP, stop_threshold: float, discount_factor: float) -> DiscretePolicy:
    """
    Value iteration algorithm. Alternate implementation with finer granularity of states evaluation
    """
    V_array = np.zeros_like(env.states_array, dtype=float)

    while True:
        delta = 0

        for s, V in enumerate(V_array):
            prev_V = V
            Q_at_s = ((env.p[:, s, :].T * discount_factor * V_array).T + env.r[:, s, :] * env.p[:, s, :]).sum(0)
            V = max(Q_at_s)
            V_array[s] = V
            delta = max(delta, np.abs(prev_V - V))

        if delta < stop_threshold:
            break

    Q = V2Q(V_array, env)
    policy = make_greedy_policy(Q)

    return policy
