#!/usr/bin/env python3

import itertools
import logging
import sys

import numpy as np
from scipy.stats import linregress


class Learner:
    def __init__(self, env, q=None, epsilon=0.0, alpha=0.1, gamma=1.0, steps=1, render_each=None):
        self.env = env
        if q is None:
            self.q = np.zeros((env.states, env.actions), dtype=np.float)
        else:
            self.q = q
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.steps = steps
        self.window_state_actions = np.zeros((self.steps, 2), dtype=np.uint)
        self.window_rewards = np.zeros(self.steps, dtype=np.float)
        self.window_multipliers = [np.power(self.gamma, np.roll(np.arange(self.steps), i)) for i in range(self.steps)]
        self.gamma_to_steps = np.power(self.gamma, self.steps)
        self.render_each = render_each

    def learn_from_trajectory(self):
        initial_state, trajectory = self.env.expert_trajectory()
        g = 0
        for i in range(len(trajectory) - 1, -1, -1):
            if i >= 1:
                state = trajectory[i - 1][2]
            else:
                state = initial_state
            action = trajectory[i][0]
            reward = trajectory[i][1]
            g = self.gamma * g + reward
            self.q[state, action] += self.alpha * (g - self.q[state, action])

    def perform(self, train=True, evaluate=False, episodes=None, stats_plot_each=None, window_size=100):
        if stats_plot_each is not None:
            try:
                from livelossplot import PlotLosses

                liveplot = PlotLosses()
            except ModuleNotFoundError:
                pass
        episode_rewards = []
        trend = 0
        try:
            for episode in itertools.count():
                if episodes is not None and episode >= episodes:
                    break
                episode_reward = self.perform_episode(train, evaluate)
                episode_rewards.append(episode_reward)
                if len(episode_rewards) >= window_size / 2:
                    trend, _, _, _, _ = linregress(np.arange(min(window_size, len(episode_rewards))),
                                                   episode_rewards[-window_size:])
                    if len(episode_rewards) >= window_size and trend < 0.01:
                        self.epsilon *= 0.99
                try:
                    liveplot.update(
                        {'reward': episode_reward,
                         'epsilon': self.epsilon,
                         'reward mean': np.mean(episode_rewards[-window_size:]),
                         'reward std': np.std(episode_rewards[-window_size:]),
                         'reward trend': trend,
                         'non-zero state-actions': np.count_nonzero(self.q)})
                    if self.env.episode and self.env.episode % stats_plot_each == 0:
                        liveplot.draw()
                except UnboundLocalError:
                    pass
            assert len(episode_rewards) == episodes
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
        step_done = None
        for step in itertools.count():
            if done and step_done is None:
                step_done = step
            if step_done is not None and step >= step_done + self.steps - 1:
                break
            if not done:
                if self.render_each and self.env.episode and self.env.episode % self.render_each == 0:
                    self.env.render()
                # Epsilon-greedy policy
                if np.random.random_sample() < self.epsilon:
                    action = np.random.randint(self.env.actions)
                else:
                    action = np.argmax(self.q[state, :])
                next_state, reward, done, _ = self.env.step(action)
                self.window_state_actions[step % self.steps] = (state, action)
                self.window_rewards[step % self.steps] = reward
                episode_reward += reward
            else:
                next_state = None
                self.window_rewards[step % self.steps] = 0
            if train and step >= self.steps - 1:
                if not done:
                    tail_reward = self.gamma_to_steps * np.max(self.q[next_state, :])
                else:
                    tail_reward = 0
                g = (self.window_rewards * self.window_multipliers[(step + 1) % self.steps]).sum() + tail_reward
                state_0 = self.window_state_actions[(step + 1) % self.steps][0]
                action_0 = self.window_state_actions[(step + 1) % self.steps][1]
                # TODO: Add support for double Q-learning.
                self.q[state_0, action_0] += self.alpha * (g - self.q[state_0, action_0])
            state = next_state
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
