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
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
    parser.add_argument("--alpha", default=0.5, type=float, help="Learning rate.")
    parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--input", "-i")
    parser.add_argument("--output", "-o")
    parser.add_argument("--format", default="py", choices=["py", "npy"])
    parser.add_argument("--stats_plot_each", type=int)
    parser.add_argument("--expert_trajectories", default=0, type=int,
                        help="Number of expert trajectories to learn from.")
    parser.add_argument("--steps", default=1, type=int, help="Number of steps for n-step learning.")
    args = parser.parse_args()

    q = None
    if args.input is not None:
        try:
            q = rl.load(args.input, format=args.format)
        except FileNotFoundError:
            pass

    learner = rl.Learner(lunar_lander_evaluator.environment(), q=q, epsilon=args.epsilon, alpha=args.alpha,
                         gamma=args.gamma, steps=args.steps, render_each=args.render_each)

    logging.info('Beginning learning from expert trajectories.')
    try:
        import tqdm

        t = tqdm.tqdm(total=args.expert_trajectories, unit="trajectory")
    except ModuleNotFoundError:
        pass
    try:
        for _ in range(args.expert_trajectories):
            learner.learn_from_trajectory()
            try:
                t.update()
            except UnboundLocalError:
                pass
    finally:
        try:
            t.close()
        except UnboundLocalError:
            pass
        if args.output is not None:
            rl.save(args.output, learner.q, format=args.format)

    logging.info('Beginning training.')
    try:
        learner.perform(train=True, evaluate=False, episodes=args.episodes, stats_plot_each=args.stats_plot_each)
    finally:
        if args.output is not None:
            rl.save(args.output, learner.q, format=args.format)

    logging.info('Beginning evaluation.')
    learner.epsilon = 0.0
    learner.perform(train=False, evaluate=True, episodes=100)
