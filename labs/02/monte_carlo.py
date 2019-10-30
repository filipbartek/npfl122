#!/usr/bin/env python3

import logging
import math
import time

import cart_pole_evaluator
import numpy as np


def main(args):
    # Create the environment
    env = cart_pole_evaluator.environment()
    pi = np.zeros(env.states, dtype=np.int)
    q = np.zeros((env.states, env.actions), dtype=np.float)
    n = np.zeros((env.states, env.actions), dtype=np.uint)
    training = True
    training_episode_i = 0
    returns = []

    try:
        from livelossplot import PlotLosses
        liveplot = PlotLosses()
    except ModuleNotFoundError:
        pass

    while training:
        if args.episodes is not None and training_episode_i >= args.episodes:
            break

        time_episode_start = time.time()

        epsilon = current_epsilon(args, training_episode_i)
        episode = generate_episode(env, pi, epsilon,
                                   args.render_each and training_episode_i % args.render_each == 0)
        improve_policy(pi, q, n, episode, args.gamma)

        returns.append(len(episode))
        assert len(returns[-100:]) <= 100
        mean_return = np.mean(np.array(returns[-100:]))
        if mean_return >= 495 and len(episode) == 500:
            logging.info('Converged.')
            training = False

        if 'liveplot' in locals():
            liveplot.update({'return': len(episode), 'mean_return': mean_return, 'epsilon': epsilon,
                             'time_per_episode': (time.time() - time_episode_start),
                             'state_actions_explored': np.count_nonzero(n)})
            if training_episode_i % 100 == 0 or (
                    args.episodes is not None and training_episode_i == args.episodes - 1) or not training:
                liveplot.draw()

        training_episode_i += 1

    # Perform last 100 evaluation episodes
    start_evaluate = True

    # Run 100 episodes for evaluation.
    # Stop exploring during evaluation.
    for evaluation_episode_i in range(100):
        generate_episode(env, pi, 0.0, args.render_each and env.episode and env.episode % args.render_each == 0)


def current_epsilon(args, episode_i):
    epsilon = args.epsilon
    if args.episodes is not None and args.epsilon_final is not None:
        progress = episode_i / args.episodes
        epsilon = (1 - progress) * args.epsilon + progress * args.epsilon_final
    epsilon *= math.pow(args.epsilon_decay, episode_i)
    return epsilon


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
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
    parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
    parser.add_argument("--epsilon_decay", default=0.99, type=float, help="")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    # Gamma smaller than 1 may diminish the effect of fixed episode length, that may cause lower score for states far from the initial.
    # --render_each=20
    args = parser.parse_args()

    main(args)
