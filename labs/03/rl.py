#!/usr/bin/env python3

import logging
import sys

import numpy as np


def perform_episode(env, q, train=True, evaluate=False, epsilon=0.0, alpha=0.1, gamma=1.0, render_each=None):
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
    return episode_reward


def perform(env, q=None, train=True, evaluate=False, epsilon=0.0, alpha=0.1, gamma=1.0, episodes=None,
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
    try:
        while episodes is None or env.episode < episodes:
            episode_reward = perform_episode(env, q, train, evaluate, epsilon, alpha, gamma, render_each)
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
    finally:
        try:
            if liveplot.global_step >= 1:
                liveplot.draw()
        except UnboundLocalError:
            pass
    return q, episode_rewards


def save(file, model, format='py'):
    success = False
    if format == 'py':
        with open(file, 'w') as f:
            # https://stackoverflow.com/a/57891767/4054250
            options = np.get_printoptions()
            np.set_printoptions(threshold=sys.maxsize)
            f.write(repr(model))
            np.set_printoptions(**options)
            success = True
    if format == 'npy':
        np.save(file, model)
        success = True
    assert success
    logging.info(f'Model saved into "{file}".')


def load(file, format='py'):
    model = None
    if format == 'py':
        with open(file) as f:
            from numpy import array
            model = eval(f.read())
    if format == 'npy':
        model = np.load(file)
    assert model is not None
    logging.info(f'Model loaded from "{file}".')
    return model
