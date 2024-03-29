import numpy as np
from envs.discrete_MDP import DiscreteMDP
from policy import DiscretePolicy, make_epsilon_greedy_policy, make_greedy_policy
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


def calculate_n_step_return(episode: list, V: np.ndarray, t: int, n_steps: int, discount_factor: float):
    """
    Calculate n-step return for episode
    """
    n_step_return = 0
    T = len(episode)

    for i in range(0, min(n_steps + 1, T - t)):
        (s, _, r) = episode[t + i]

        if i == n_steps:
            n_step_return += discount_factor ** i * V[s]
        else:
            n_step_return += discount_factor ** i * r

    return n_step_return


def n_step_td_prediction(env: DiscreteMDP, policy: DiscretePolicy, n_steps: int, n_episodes: int, step_size: float,
                         discount_factor: float = 1.0):
    """
    n-step TD prediction algorithm
    """
    assert n_steps > 0

    V = np.zeros(env.states_num)

    for _ in range(n_episodes):
        episode = sample_episode(env, policy)

        for t, (s, _, _) in enumerate(episode):
            n_step_return = calculate_n_step_return(episode, V, t, n_steps, discount_factor)
            V[s] += step_size * (n_step_return - V[s])

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


def calculate_n_step_action_value_return(episode: list, Q: np.ndarray, t: int, n_steps: int, discount_factor: float):
    """
    Calculate n-step return for episode (for action-value function)
    """
    n_step_return = 0
    T = len(episode)

    for i in range(0, min(n_steps + 1, T - t)):
        (s, a, r) = episode[t + i]

        if i == n_steps:
            n_step_return += discount_factor ** i * Q[s, a]
        else:
            n_step_return += discount_factor ** i * r

    return n_step_return


def n_step_sarsa(env: DiscreteMDP, n_steps: int, n_episodes: int, step_size: float, discount_factor: float = 1.0,
                 eps: float = 0.1):
    """
    SARSA algorithm (on-policy n-step TD control)
    """
    assert n_steps > 0

    Q = np.zeros([env.states_num, env.actions_num])
    sampler = EpisodeSampler(env)
    policy = make_epsilon_greedy_policy(Q, eps)

    for _ in range(n_episodes):
        sampler.reset()

        done = False
        episode = []

        for _ in range(n_steps):
            state, action, reward, done = sampler.sample_one_step(policy)
            episode.append((state, action, reward))
            if done:
                break

        t = 0
        while t < len(episode):
            if not done:
                next_state, next_action, next_reward, done = sampler.sample_one_step(policy)
                episode.append((next_state, next_action, next_reward))

            n_step_return = calculate_n_step_action_value_return(episode, Q, t, n_steps, discount_factor)
            (state, action, _) = episode[t]

            Q[state, action] += step_size * (n_step_return - Q[state, action])

            policy = make_epsilon_greedy_policy(Q, eps)
            t += 1

    return policy, Q


def calculate_importance_sampling_ratio(episode: list, t: int, n_steps: int, behavior_policy: DiscretePolicy,
                                        target_policy: DiscretePolicy):
    """
    Calculate importance sampling ratio
    """
    w = 1
    for i in range(t + 1, min(t + n_steps, len(episode))):
        (s, a, _) = episode[i]
        w *= target_policy.p[s, a] / behavior_policy.p[s, a]

    return w


def off_policy_n_step_sarsa(env: DiscreteMDP, behaviour_policy: DiscretePolicy, n_steps: int, n_episodes: int,
                            step_size: float, discount_factor: float = 1.0):
    """
    Off-policy SARSA algorithm (on-policy n-step TD control)
    """
    assert n_steps > 0

    Q = np.zeros([env.states_num, env.actions_num])
    target_policy = make_greedy_policy(Q)

    for _ in range(n_episodes):
        episode = sample_episode(env, behaviour_policy)

        for t in range(len(episode)):
            n_step_return = calculate_n_step_action_value_return(episode, Q, t, n_steps, discount_factor)
            importance_sampling_ratio = calculate_importance_sampling_ratio(episode, t, n_steps, behaviour_policy,
                                                                            target_policy)
            (state, action, _) = episode[t]

            Q[state, action] += step_size * importance_sampling_ratio * (n_step_return - Q[state, action])

            target_policy = make_greedy_policy(Q)

    return target_policy, Q


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
