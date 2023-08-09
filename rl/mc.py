import gymnasium as gym
import numpy as np
from envs.discrete_MDP import DiscreteMDP
from policy import Policy, DiscretePolicy, make_greedy_policy


def sample_episode(env: gym.Env, policy: Policy):
    """
    Sample episode from environment with given policy

    Return list of (state, action, reward) tuples.
    """
    episode = []
    terminated = False
    state, _ = env.reset()

    while not terminated:
        action = policy.sample_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        if reward is None:
            breakpoint()
        episode.append((state, action, float(reward)))
        state = next_state

    return episode


def calculate_discounted_returns(episode: list, discount_factor: float):
    """
    Calculate discounted returns for episode
    """
    discounted_returns = np.zeros(len(episode))
    discounted_return = 0
    for i, (_, _, r) in enumerate(reversed(episode)):
        discounted_return = r + discount_factor * discounted_return
        discounted_returns[-i-1] = discounted_return

    return discounted_returns


def mc_first_visit_policy_eval(env: DiscreteMDP, policy: DiscretePolicy, n_episodes: int, discount_factor: float = 1.0):
    """
    Monte Carlo first-visit policy evaluation algorithm
    """
    returns = np.zeros(env.states_num)
    n_visits = np.zeros(env.states_num)

    for _ in range(n_episodes):
        episode = sample_episode(env, policy)

        discounted_returns = calculate_discounted_returns(episode, discount_factor)

        state_visited = np.zeros(env.states_num, dtype=bool)
        for (s, _, _), discounted_return in zip(episode, discounted_returns):
            if not state_visited[s]:
                state_visited[s] = True
                returns[s] += discounted_return
                n_visits[s] += 1

    V = returns / n_visits

    return V


def mc_every_visit_policy_eval(env: DiscreteMDP, policy: DiscretePolicy, n_episodes: int, discount_factor: float = 1.0):
    """
    Monte Carlo every-visit policy evaluation algorithm
    """
    returns = np.zeros(env.states_num)
    n_visits = np.zeros(env.states_num)

    for _ in range(n_episodes):
        episode = sample_episode(env, policy)

        discounted_returns = calculate_discounted_returns(episode, discount_factor)

        for (s, _, _), discounted_return in zip(episode, discounted_returns):
            returns[s] += discounted_return
            n_visits[s] += 1

    V = returns / n_visits

    return V


def mc_first_visit_eps_greedy_control(env: DiscreteMDP, n_episodes: int, eps: float, discount_factor: float = 1.0):
    """
    Monte Carlo first-visit epsilon-greedy control algorithm
    """
    Q = np.zeros((env.states_num, env.actions_num))
    n_visits = np.zeros_like(Q)

    policy = DiscretePolicy.from_env(env)
    policy.p = np.ones_like(policy.p) / env.actions_num

    for _ in range(n_episodes):
        episode = sample_episode(env, policy)

        discounted_returns = calculate_discounted_returns(episode, discount_factor)

        state_action_visited = np.zeros((env.states_num, env.actions_num), dtype=bool)
        for (s, a, _), discounted_return in zip(episode, discounted_returns):
            if not state_action_visited[s, a]:
                state_action_visited[s, a] = True
                Q[s, a] = (Q[s, a] * n_visits[s, a] + discounted_return) / (n_visits[s, a] + 1)
                n_visits[s, a] += 1
                best_action = np.argmax(Q[s])
                policy.p[s, :] = eps / env.actions_num
                policy.p[s, best_action] += 1 - eps

    return policy, Q


def calculate_importance_sampling_ratio(episode: list, behavior_policy: DiscretePolicy, target_policy: DiscretePolicy):
    """
    Calculate importance sampling ratio for each step of episode
    """
    W = np.ones(len(episode))
    w = 1
    for i, (s, a, _) in enumerate(reversed(episode)):
        w *= target_policy.p[s, a] / behavior_policy.p[s, a]
        W[-i-1] = w

    return W


def mc_off_policy_prediction(env: DiscreteMDP, behavior_policy: DiscretePolicy, target_policy: DiscretePolicy,
                             n_episodes: int, discount_factor: float = 1.0):
    """
    Monte Carlo off-policy prediction algorithm
    """
    V = np.zeros(env.states_num)
    C = np.zeros_like(V)

    for _ in range(n_episodes):
        episode = sample_episode(env, behavior_policy)

        discounted_returns = calculate_discounted_returns(episode, discount_factor)
        importance_sampling_ratio = calculate_importance_sampling_ratio(episode, behavior_policy, target_policy)

        for (s, _, _), G, W in zip(episode, discounted_returns, importance_sampling_ratio):
            if W == 0:
                continue
            C[s] += W
            V[s] += W * (G - V[s]) / C[s]

    return V


# def mc_off_policy_prediction(env: DiscreteMDP, behavior_policy: DiscretePolicy, target_policy: DiscretePolicy,
#                              n_episodes: int, discount_factor: float = 1.0):
#     """
#     Monte Carlo off-policy prediction algorithm
#     """
#     Q = np.zeros((env.states_num, env.actions_num))
#     C = np.zeros_like(Q)
#
#     for _ in range(n_episodes):
#         episode = sample_episode(env, behavior_policy)
#
#         discounted_returns = calculate_discounted_returns(episode, discount_factor)
#         importance_sampling_ratio = calculate_importance_sampling_ratio(episode, behavior_policy, target_policy)
#
#         for (s, a, _), G, W in zip(episode, discounted_returns, importance_sampling_ratio):
#             if W == 0:
#                 continue
#             C[s, a] += W
#             Q[s, a] += W * (G - Q[s, a]) / C[s, a]
#
#     return Q
