#!/usr/bin/env python3

import concurrent.futures
import datetime
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from scipy.stats import linregress
from tqdm import tqdm

import mountain_car_evaluator


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


def perform_batch_isolated(evaluator, tiles, W=None, alpha=0.0, epsilon=0.0, gamma=1.0, episodes=1, evaluating=False):
    env = evaluator.environment(tiles=tiles)
    if W is None:
        W = np.zeros((env.weights, env.actions))
    else:
        W = W.copy()
    rewards = list()
    for _ in range(episodes):
        rewards.append(perform_episode(env, W, alpha, epsilon, gamma, evaluating))
    return rewards, W, alpha, epsilon, gamma


def perform_episode(env, W, alpha, epsilon, gamma, evaluating=False):
    # Perform a training episode
    episode_reward = 0
    state, done = env.reset(evaluating), False
    state_x = feature_vector(state, env)
    while not done:
        if args.render_each and env.episode and env.episode % args.render_each == 0:
            env.render()

        # Choose `action` according to epsilon-greedy strategy
        if np.random.random_sample() < epsilon:
            action = np.random.randint(env.actions)
        else:
            action = np.argmax(action_preferences(state_x, env, W))

        next_state, reward, done, _ = env.step(action)
        next_state_x = feature_vector(next_state, env)
        episode_reward += reward

        # Update W values
        g = reward + gamma * np.max(action_preferences(next_state_x, env, W))
        W[:, action] += alpha * (g - np.max(action_preferences(state_x, env, W))) * state_x

        state_x = next_state_x
    return episode_reward


def feature_vector(state, env):
    assert len(state) == args.tiles
    assert all(0 <= value < env.weights for value in state)
    assert len(state) == len(set(state))
    x = np.zeros(env.weights, dtype=np.bool)
    x[state] = True
    return x


def action_preferences(state_x, env, W):
    assert state_x.shape == (env.weights,)
    return np.dot(state_x.transpose(), W)


def train_mt(args, W, summary_writer_train):
    alpha = args.alpha / args.tiles
    epsilon = args.epsilon
    gamma = args.gamma
    with ThreadPoolExecutor(max_workers=args.cpus) as executor, tqdm(total=args.episodes, unit='episode') as t:
        futures = set()
        episodes_submitted = 0
        batch_i = 0
        while batch_i * args.episodes_per_batch < args.episodes:
            while len(futures) < args.cpus and episodes_submitted < args.episodes:
                if args.epsilon_final is not None:
                    epsilon = np.exp(
                        np.interp(episodes_submitted, [0, args.episodes],
                                  [np.log(args.epsilon), np.log(args.epsilon_final)]))
                if args.alpha_final is not None:
                    alpha = np.exp(np.interp(episodes_submitted, [0, args.episodes],
                                             [np.log(args.alpha), np.log(args.alpha_final)])) / args.tiles
                future = executor.submit(perform_batch_isolated, mountain_car_evaluator, args.tiles, W, alpha,
                                         epsilon,
                                         gamma, args.episodes_per_batch)
                futures.add(future)
                episodes_submitted += args.episodes_per_batch
            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                assert future.done()
                rewards, W_new, alpha, epsilon, gamma = future.result()
                if W is None:
                    W = np.zeros(W_new.shape)
                W = np.average((W, W_new), axis=0, weights=(0.5, 0.5))
                step = batch_i * args.episodes_per_batch
                with summary_writer_train.as_default():
                    tf.summary.scalar('alpha', alpha, step=step)
                    tf.summary.scalar('epsilon', epsilon, step=step)
                    tf.summary.scalar('gamma', gamma, step=step)
                    tf.summary.scalar('episodes_per_batch', args.episodes_per_batch, step=step)
                    tf.summary.histogram('reward[batch]', rewards, step=env.episode)
                    tf.summary.scalar('reward[batch].mean', np.mean(rewards), step=step)
                    tf.summary.histogram('W', W, step=step)
                    tf.summary.scalar('W.nonzero', np.count_nonzero(W), step=step)
                    tf.summary.scalar('W.nonzero.ratio', np.count_nonzero(W) / W.size, step=step)
                t.update(args.episodes_per_batch)
                batch_i += 1
    return W


def train_st(args, W, summary_writer_train):
    assert args.episodes is not None
    env = mountain_car_evaluator.environment(tiles=args.tiles)
    if W is None:
        W = np.zeros((env.weights, env.actions))
    alpha = args.alpha / args.tiles
    epsilon = args.epsilon
    gamma = args.gamma
    window_size = 100
    episode_rewards = deque(maxlen=window_size)
    trend = None
    for _ in tqdm(range(args.episodes), unit='episode'):
        if args.episodes is not None and env.episode >= args.episodes:
            break

        if args.epsilon_final is not None:
            epsilon = np.exp(
                np.interp(env.episode, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
        if args.alpha_final is not None:
            alpha = np.exp(np.interp(env.episode, [0, args.episodes],
                                     [np.log(args.alpha), np.log(args.alpha_final)])) / args.tiles

        episode_reward = perform_episode(env, W, alpha, epsilon, gamma)
        episode_rewards.append(episode_reward)
        if len(episode_rewards) == window_size:
            trend, _, _, _, _ = linregress(np.arange(window_size), episode_rewards)
            # trend <= 0: epsilon *= 0.9 (mindecay)
            # trend = 1: epsilon *= 1
            # trend >= 2: epsilon *= 1.1 (maxdecay)
            # if trend < 0.01:
            #    epsilon *= 0.95

        with summary_writer_train.as_default():
            tf.summary.scalar('reward', episode_reward, step=env.episode)
            tf.summary.scalar('epsilon', epsilon, step=env.episode)
            tf.summary.scalar('alpha', alpha, step=env.episode)
            tf.summary.scalar('gamma', gamma, step=env.episode)
            tf.summary.scalar('W.nonzero', np.count_nonzero(W), step=env.episode)
            tf.summary.scalar('W.nonzero.ratio', np.count_nonzero(W) / W.size, step=env.episode)
            tf.summary.histogram('W', W, step=env.episode)
            if len(episode_rewards) >= window_size:
                tf.summary.histogram(f'reward[{window_size}]', episode_rewards, step=env.episode)
            if trend is not None:
                tf.summary.scalar(f'reward[{window_size}].trend', trend, step=env.episode)
    return W


if __name__ == "__main__":
    np.seterr(all='raise')

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
    parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.1, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.001, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--input", "-i", default="q_learning_tiles_model.py")
    parser.add_argument("--episodes_per_batch", default=1, type=int)
    parser.add_argument("--no_evaluation", action="store_true")
    parser.add_argument("--cpus", default=1, type=int)
    args = parser.parse_args()

    try:
        W = load_py(args.input)
    except FileNotFoundError:
        W = None
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer_train = tf.summary.create_file_writer(os.path.join(args.logdir, run_name, 'train'))

    try:
        if args.cpus > 1:
            train_mt(args, W, summary_writer_train)
        else:
            train_st(args, W, summary_writer_train)
    finally:
        assert W is not None
        save_py('q_learning_tiles_' + run_name + '.py', W)

    if not args.no_evaluation:
        perform_batch_isolated(mountain_car_evaluator, args.tiles, W, gamma=args.gamma, episodes=100, evaluating=True)
