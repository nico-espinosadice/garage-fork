# Copied from stable_baselines
import numpy as np

from stable_baselines3.common.vec_env import VecEnv


def evaluate_policy(
    model,
    env,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    callback=None,
    reward_threshold=None,
    return_lengths=False,
    return_actions=False,
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseAlgorithm) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths, episode_goal_achieved, actions = [], [], [], []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            actions.append(action)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if "goal_achieved" in _info:
            episode_goal_achieved.append(_info['goal_achieved'])
        else:
            episode_goal_achieved.append(False)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_goals = np.mean(episode_goal_achieved)
    std_goals = np.std(episode_goal_achieved)
    mean_episode_lengths = np.mean(episode_lengths)
    std_episode_lengths = np.std(episode_lengths)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_actions:
        return mean_reward, std_reward, mean_goals, std_goals, mean_episode_lengths, std_episode_lengths, actions
    elif return_lengths:
        return mean_reward, std_reward, mean_goals, std_goals, mean_episode_lengths, std_episode_lengths
    else:
        return mean_reward, std_reward