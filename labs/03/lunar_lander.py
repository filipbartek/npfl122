#!/usr/bin/env python3

# Team members:
# [Filip Bartek](https://recodex.mff.cuni.cz/app/user/9d1ef2af-eb87-11e9-9ce9-00505601122b)

import logging

import numpy as np

import lunar_lander_evaluator
import rl

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=None, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--stats_plot_each", type=int)
    args = parser.parse_args()

    # Create the environment
    env = lunar_lander_evaluator.environment()

    logging.info('Beginning training.')
    q, _ = rl.perform(env, train=True, evaluate=False, episodes=args.episodes, epsilon=args.epsilon, alpha=args.alpha,
                      gamma=args.gamma, render_each=args.render_each, stats_plot_each=args.stats_plot_each)

    logging.info('Beginning evaluation.')
    rl.perform(env, train=False, evaluate=True, episodes=100, gamma=args.gamma, render_each=args.render_each)
