#!/usr/bin/env python3

import logging
import sys
import time

import cart_pole_evaluator
import numpy as np
import scipy.stats


def main(args):
    # Create the environment
    env = cart_pole_evaluator.environment()

    try:
        with open(args.model) as pi_file:
            from numpy import array
            pi = eval(pi_file.read())
    except FileNotFoundError:
        pi = train_pi(args, env)
        with open(args.model, 'w') as pi_file:
            # https://stackoverflow.com/a/57891767/4054250
            options = np.get_printoptions()
            np.set_printoptions(threshold=sys.maxsize)
            pi_file.write(repr(pi))
            np.set_printoptions(**options)

    evaluate_pi(env, pi, args.render_each)


def evaluate_pi(env, pi, render_each):
    # Perform last 100 evaluation episodes
    start_evaluate = True

    # Run 100 episodes for evaluation.
    # Stop exploring during evaluation.
    for evaluation_episode_i in range(100):
        generate_episode(env, pi, 0.0, render_each and env.episode and env.episode % render_each == 0)


def train_pi(args, env):
    try:
        from livelossplot import PlotLosses
        liveplot = PlotLosses()
    except ModuleNotFoundError:
        pass

    pi = np.zeros(env.states, dtype=np.int)
    q = np.zeros((env.states, env.actions), dtype=np.float)
    n = np.zeros((env.states, env.actions), dtype=np.uint)
    training = True
    training_episode_i = 0
    returns = []
    epsilon = args.epsilon
    window_size = 100
    slope = 0

    while training:
        if args.episodes is not None and training_episode_i >= args.episodes:
            break

        time_episode_start = time.time()

        episode = generate_episode(env, pi, epsilon,
                                   args.render_each and training_episode_i % args.render_each == 0)
        improve_policy(pi, q, n, episode, args.gamma)

        returns.append(len(episode))
        if len(returns) >= window_size / 2:
            slope, _, _, _, _ = scipy.stats.linregress(np.arange(min(window_size, len(returns))),
                                                       returns[-window_size:])
            if len(returns) >= window_size and slope < 0.01:
                epsilon *= 0.95
        window_mean = np.mean(returns[-window_size:])
        window_std = np.std(returns[-window_size:])
        if len(returns) >= window_size and window_mean >= 498 and window_std <= 2 and len(episode) >= 500:
            logging.info('Converged.')
            training = False

        if 'liveplot' in locals():
            log = {'return': len(episode), '100_returns_mean': window_mean, '100_returns_std': window_std,
                   'epsilon': epsilon, 'state_actions_visited': np.count_nonzero(n), 'slope': slope}
            liveplot.update(log)
            if training_episode_i % window_size == 0 or (
                    args.episodes is not None and training_episode_i == args.episodes - 1) or not training:
                liveplot.draw()

        training_episode_i += 1

    return pi


def generate_episode(env, pi, epsilon=0.0, render=False):
    state, done = env.reset(), False
    # There are 500 steps in each episode. Consider optimizing by preallocating.
    episode = []
    while not done:
        if render:
            env.render()
        if np.random.random_sample() < epsilon:
            action = np.random.randint(env.actions)
        else:
            action = pi[state]
        new_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        state = new_state
    return episode


def improve_policy(pi, q, n, episode, gamma):
    g = 0
    for state, action, reward in reversed(episode):
        g = gamma * g + reward
        n[state, action] += 1
        q[state, action] += (g - q[state, action]) / n[state, action]
        pi[state] = np.argmax(q[state, :])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="monte_carlo_pi.py", help="Model Python file")
    parser.add_argument("--episodes", default=2000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
    parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    # Gamma smaller than 1 may diminish the effect of fixed episode length
    # (lower score for states far from the initial).
    args = parser.parse_args()

    main(args)
