import numpy as np
from envs.discrete_MDP import DiscreteMDP
from policy import DiscretePolicy, make_epsilon_greedy_policy
from rl.utils import sample_episode, EpisodeSampler


def td_prediction(env: DiscreteMDP, policy: DiscretePolicy, n_episodes: int, step_size: float,
                  discount_factor: float = 1.0):
    """
    TD(0) prediction algorithm
    """
    V = np.zeros(env.states_num)

    for _ in range(n_episodes):
        episode = sample_episode(env, policy)

        for (s, _, r), (s_next, _, _) in zip(episode, episode[1:]):
            V[s] += step_size * (r + discount_factor * V[s_next] - V[s])

        (s, _, r) = episode[-1]
        V[s] += step_size * (r - V[s])

    return V


def sarsa(env: DiscreteMDP, n_episodes: int, step_size: float, discount_factor: float = 1.0, eps: float = 0.1):
    """
    SARSA algorithm (on-policy one-step TD control)
    """
    Q = np.zeros([env.states_num, env.actions_num])
    sampler = EpisodeSampler(env)

    for _ in range(n_episodes):
        sampler.reset()

        done = False
        state = None
        action = None
        reward = None

        while not done:
            policy = make_epsilon_greedy_policy(Q, eps)
            next_state, next_action, next_reward, done = sampler.sample_one_step(policy)

            if state is not None:
                Q[state, action] += step_size * (reward + discount_factor * Q[next_state, next_action] - Q[state, action])

            state = next_state
            action = next_action
            reward = next_reward

        Q[state, action] += step_size * (reward - Q[state, action])

    policy = make_epsilon_greedy_policy(Q, eps)

    return policy, Q


def expected_sarsa(env: DiscreteMDP, n_episodes: int, step_size: float, discount_factor: float = 1.0, eps: float = 0.1):
    """
    Expected SARSA algorithm (on-policy one-step TD control)
    """
    Q = np.zeros([env.states_num, env.actions_num])
    sampler = EpisodeSampler(env)

    for _ in range(n_episodes):
        sampler.reset()

        done = False
        state = None
        action = None
        reward = None

        while not done:
            policy = make_epsilon_greedy_policy(Q, eps)
            next_state, next_action, next_reward, done = sampler.sample_one_step(policy)

            if state is not None:
                Q[state, action] += step_size * (reward + discount_factor *
                                                 np.dot(policy.p[next_state, :], Q[next_state, :]) - Q[state, action])

            state = next_state
            action = next_action
            reward = next_reward

        Q[state, action] += step_size * (reward - Q[state, action])

    policy = make_epsilon_greedy_policy(Q, eps)

    return policy, Q


def Q_learning(env: DiscreteMDP, n_episodes: int, step_size: float, discount_factor: float = 1.0, eps: float = 0.1):
    """
    Q-learning algorithm (off-policy one-step TD control)
    """
    Q = np.zeros([env.states_num, env.actions_num])
    sampler = EpisodeSampler(env)

    for _ in range(n_episodes):
        sampler.reset()

        done = False
        state = None
        action = None
        reward = None

        while not done:
            policy = make_epsilon_greedy_policy(Q, eps)
            next_state, next_action, next_reward, done = sampler.sample_one_step(policy)

            if state is not None:
                Q[state, action] += step_size * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])

            state = next_state
            action = next_action
            reward = next_reward

        Q[state, action] += step_size * (reward - Q[state, action])

    policy = make_epsilon_greedy_policy(Q, eps)

    return policy, Q


def double_Q_learning(env: DiscreteMDP, n_episodes: int, step_size: float, discount_factor: float = 1.0, eps: float = 0.1):
    """
    Double Q-learning algorithm (off-policy one-step TD control)
    """
    Q1 = np.zeros([env.states_num, env.actions_num])
    Q2 = np.zeros([env.states_num, env.actions_num])
    sampler = EpisodeSampler(env)

    for _ in range(n_episodes):
        sampler.reset()

        done = False
        state = None
        action = None
        reward = None

        while not done:
            policy = make_epsilon_greedy_policy(Q1 + Q2, eps)
            next_state, next_action, next_reward, done = sampler.sample_one_step(policy)

            if np.random.rand() < 0.5:
                Q = Q1
                Q_val_estimate = Q2
            else:
                Q = Q2
                Q_val_estimate = Q1

            if state is not None:
                Q[state, action] += step_size * (reward + discount_factor *
                                                 Q_val_estimate[next_state, np.argmax(Q[next_state, :])] - Q[state, action])

            state = next_state
            action = next_action
            reward = next_reward

        Q[state, action] += step_size * (reward - Q[state, action])

    policy = make_epsilon_greedy_policy(Q, eps)

    return policy, Q
