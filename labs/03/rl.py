#!/usr/bin/env python3

from collections import deque
import datetime
import itertools
import logging
import os.path
import sys

import numpy as np
import tensorflow as tf
from scipy.stats import linregress


class Learner:
    def __init__(self, evaluator, q=None, epsilon=0.0, alpha=0.1, gamma=1.0, steps=1, render_each=None, log_dir=None,
                 validation_period=None, window_size=100):
        self.evaluator = evaluator
        self.q = q
        self.best_q = None
        self.best_score = None
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.steps = steps
        self.window_state_actions = np.zeros((self.steps, 2), dtype=np.uint)
        self.window_rewards = np.zeros(self.steps, dtype=np.float)
        self.window_multipliers = np.array(
            [np.power(self.gamma, np.roll(np.arange(self.steps), i)) for i in range(self.steps)], dtype=np.float)
        self.gamma_to_steps = np.power(self.gamma, self.steps)
        self.render_each = render_each
        if log_dir is None:
            self.log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            self.log_dir = log_dir
        self.validation_period = validation_period
        self.window_size = window_size

    def environment(self):
        env = self.evaluator.environment()
        if self.q is None:
            self.q = np.zeros((env.states, env.actions), dtype=np.float)
        assert self.q.shape == (env.states, env.actions)
        if self.best_q is None:
            self.best_q = self.q.copy()
        return env

    def learn_from_trajectory(self, env):
        initial_state, trajectory = env.expert_trajectory()
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
        return g, len(trajectory)

    def validate(self, step=0, episodes=100, evaluate=False):
        assert episodes >= 1
        env = self.environment()
        rewards = np.zeros(episodes, dtype=np.float)
        episode_lengths = np.zeros(episodes, dtype=np.uint)
        for episode in range(episodes):
            episode_reward, episode_length = self.perform_episode(env=env, validate=True, evaluate=evaluate)
            rewards[episode] = episode_reward
            episode_lengths[episode] = episode_length
        if evaluate:
            summary_name = 'evaluate'
        else:
            summary_name = 'validate'
        with tf.summary.create_file_writer(os.path.join(self.log_dir, summary_name)).as_default():
            tf.summary.scalar('gamma', self.gamma, step=step)
            tf.summary.histogram(f'reward[{episodes}]', rewards, step=step)
            tf.summary.scalar(f'reward[{episodes}].mean', rewards.mean(), step=step)
            tf.summary.histogram(f'episode_length[{episodes}]', episode_lengths, step=step)
            tf.summary.scalar(f'episode_length[{episodes}].mean', episode_lengths.mean(), step=step)
        score = rewards.mean()
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_q = self.q.copy()

    def train(self, episodes=None, expert_trajectories=0):
        if episodes and episodes <= 0:
            return
        logging.info('Beginning training.')
        summary_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'train'))
        episode_rewards = deque(maxlen=self.window_size)
        episode_lengths = deque(maxlen=self.window_size)
        trend = None
        env = self.environment()
        episode = 0
        try:
            while True:
                if self.validation_period and episode % self.validation_period == 0:
                    self.validate(step=episode, episodes=self.window_size)
                if episodes is not None and episode >= episodes:
                    break
                if episode < expert_trajectories:
                    episode_reward, episode_length = self.learn_from_trajectory(env)
                else:
                    episode_reward, episode_length = self.perform_episode(env)
                episode += 1
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                if len(episode_rewards) >= expert_trajectories + self.window_size:
                    trend, _, _, _, _ = linregress(np.arange(min(self.window_size, len(episode_rewards))), episode_rewards)
                    if trend < 0.01:
                        self.epsilon *= 0.99
                with summary_writer.as_default():
                    if episode == expert_trajectories:
                        tf.summary.text('info', 'This is the last episode trained on an expert trajectory.', step=episode)
                    tf.summary.scalar('reward', episode_reward, step=episode)
                    tf.summary.scalar('episode_length', episode_length, step=episode)
                    tf.summary.scalar('epsilon', self.epsilon, step=episode)
                    tf.summary.scalar('alpha', self.alpha, step=episode)
                    tf.summary.scalar('gamma', self.gamma, step=episode)
                    tf.summary.scalar('steps', self.steps, step=episode)
                    tf.summary.scalar('state_actions_explored', np.count_nonzero(self.q), step=episode,
                                      description='Number of state-actions with non-zero value estimate')
                    if len(episode_rewards) >= self.window_size:
                        tf.summary.histogram(f'reward[{self.window_size}]', episode_rewards, step=episode)
                        tf.summary.scalar(f'reward[{self.window_size}].mean', np.mean(episode_rewards), step=episode,
                                          description=f'Mean reward across {self.window_size} latest episodes')
                        if trend is not None:
                            tf.summary.scalar(f'reward[{self.window_size}].trend', trend, step=episode,
                                              description=f'Reward trend across {self.window_size} latest episodes')
                        tf.summary.histogram(f'episode_length[{self.window_size}]', episode_lengths, step=episode)
                        tf.summary.scalar(f'episode_length[{self.window_size}].mean', np.mean(episode_lengths), step=episode)
        finally:
            if self.validation_period and episode % self.validation_period != 0:
                self.validate(step=episode, episodes=self.window_size)
            summary_writer.close()

    def perform_episode(self, env, validate=False, evaluate=False):
        state, done = env.reset(evaluate), False
        episode_reward = 0
        step_done = None
        for step in itertools.count():
            if done and step_done is None:
                step_done = step
                step = max(step, self.steps - 1)
            if step_done is not None and step >= step_done + self.steps - 1:
                break
            if not done:
                if self.render_each and env.episode and env.episode % self.render_each == 0:
                    env.render()
                # Epsilon-greedy policy
                if not validate and np.random.random_sample() < self.epsilon:
                    action = np.random.randint(env.actions)
                else:
                    action = np.argmax(self.q[state, :])
                next_state, reward, done, _ = env.step(action)
                self.window_state_actions[step % self.steps] = (state, action)
                self.window_rewards[step % self.steps] = reward
                episode_reward += reward
            else:
                next_state = None
                self.window_rewards[step % self.steps] = 0
            if not validate and step >= self.steps - 1:
                # Training with n-step learning
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
        return episode_reward, step_done


def save(file, model):
    get_model_format(file)['save'](file, model)
    logging.info(f'Model saved into "{file}".')


def load(file):
    model = get_model_format(file)['load'](file)
    logging.info(f'Model loaded from "{file}".')
    return model


def get_model_format(file):
    try:
        return model_formats[os.path.splitext(file)[1]]
    except KeyError as e:
        raise RuntimeError(f'Unrecognized model format: {file}') from e


def save_py(file, model):
    with open(file, 'w') as f:
        # https://stackoverflow.com/a/57891767/4054250
        options = np.get_printoptions()
        np.set_printoptions(threshold=sys.maxsize)
        f.write(repr(model))
        np.set_printoptions(**options)


def load_py(file):
    with open(file) as f:
        # noinspection PyUnresolvedReferences
        from numpy import array
        return eval(f.read())


model_formats = {
    '.py': {'save': save_py, 'load': load_py},
    '.npy': {'save': np.save, 'load': np.load}
}
