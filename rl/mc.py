import numpy as np
from envs.discrete_MDP import DiscreteMDP
from policy import DiscretePolicy, make_greedy_policy, make_epsilon_greedy_policy
from rl.utils import sample_episode


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


def mc_off_policy_control(env: DiscreteMDP, n_episodes: int, discount_factor: float = 1.0):
    """
    Monte Carlo off-policy control algorithm
    """
    Q = -1000 * np.ones((env.states_num, env.actions_num))  # FIXME -1000 for impossible actions, should be set to -inf
    C = np.zeros_like(Q)

    policy = make_greedy_policy(Q)

    for _ in range(n_episodes):
        behavior_policy = make_epsilon_greedy_policy(Q, 0.1)
        episode = sample_episode(env, behavior_policy)

        W = 1
        G = 0
        for s, a, r in reversed(episode):
            G = r + discount_factor * G
            C[s, a] += W
            Q[s, a] += W * (G - Q[s, a]) / C[s, a]

            policy.p[s, :] = 0
            policy.p[s, np.argmax(Q[s])] = 1

            if a != np.argmax(Q[s]):
                break

            W *= 1/behavior_policy.p[s, a]

    return policy
