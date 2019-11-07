#!/usr/bin/env python3

import logging
import sys

import numpy as np


class Learner:
    def __init__(self, env, q=None, epsilon=0.0, alpha=0.1, gamma=1.0, render_each=None):
        self.env = env
        if q is None:
            self.q = np.zeros((env.states, env.actions), dtype=np.float)
        else:
            self.q = q.copy()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.render_each = render_each

    def perform(self, train=True, evaluate=False, episodes=None, stats_plot_each=None, window_size=100):
        if stats_plot_each is not None:
            try:
                from livelossplot import PlotLosses

                liveplot = PlotLosses()
            except ModuleNotFoundError:
                pass
        episode_rewards = []
        try:
            while episodes is None or self.env.episode < episodes:
                episode_reward = self.perform_episode(train, evaluate)
                episode_rewards.append(episode_reward)
                try:
                    liveplot.update(
                        {'reward': episode_reward,
                         'epsilon': self.epsilon,
                         'reward mean': np.mean(episode_rewards[-window_size:]),
                         'reward std': np.std(episode_rewards[-window_size:])})
                    if self.env.episode and self.env.episode % stats_plot_each == 0:
                        liveplot.draw()
                except UnboundLocalError:
                    pass
        finally:
            try:
                if liveplot.global_step >= 1:
                    liveplot.draw()
            except UnboundLocalError:
                pass
        return episode_rewards

    def perform_episode(self, train=True, evaluate=False):
        state, done = self.env.reset(evaluate), False
        episode_reward = 0
        while not done:
            if self.render_each and self.env.episode and self.env.episode % self.render_each == 0:
                self.env.render()
            if np.random.random_sample() < self.epsilon:
                action = np.random.randint(self.env.actions)
            else:
                action = np.argmax(self.q[state, :])
            next_state, reward, done, _ = self.env.step(action)
            if train:
                self.q[state, action] += self.alpha * (
                        reward + self.gamma * np.max(self.q[next_state, :]) - self.q[state, action])
            state = next_state
            episode_reward += reward
        return episode_reward


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
