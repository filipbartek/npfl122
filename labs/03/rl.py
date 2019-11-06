#!/usr/bin/env python3

import numpy as np


def perform(env, train=True, evaluate=False, q=None, epsilon=0.0, alpha=0.1, gamma=1.0, episodes=None,
            render_each=None, stats_plot_each=None, window_size=100):
    if stats_plot_each is not None:
        try:
            from livelossplot import PlotLosses

            liveplot = PlotLosses()
        except ModuleNotFoundError:
            pass
    if q is None:
        q = np.zeros((env.states, env.actions), dtype=np.float)
    episode_rewards = []
    while episodes is None or env.episode < episodes:
        state, done = env.reset(evaluate), False
        episode_reward = 0
        while not done:
            if render_each and env.episode and env.episode % render_each == 0:
                env.render()
            if np.random.random_sample() < epsilon:
                action = np.random.randint(env.actions)
            else:
                action = np.argmax(q[state, :])
            next_state, reward, done, _ = env.step(action)
            if train:
                q[state, action] += alpha * (reward + gamma * np.max(q[next_state, :]) - q[state, action])
            state = next_state
            episode_reward += reward
        episode_rewards.append(episode_reward)
        try:
            liveplot.update(
                {'reward': episode_reward,
                 'epsilon': epsilon,
                 'reward mean': np.mean(episode_rewards[-window_size:]),
                 'reward std': np.std(episode_rewards[-window_size:])})
            if env.episode and env.episode % stats_plot_each == 0:
                liveplot.draw()
        except UnboundLocalError:
            pass
    try:
        liveplot.draw()
    except UnboundLocalError:
        pass
    return q, episode_rewards
