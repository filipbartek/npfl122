#!/usr/bin/env python3

# Team members:
# Filip Bartek       | https://recodex.mff.cuni.cz/app/user/9d1ef2af-eb87-11e9-9ce9-00505601122b
# Bartosz Piotrowski | https://recodex.mff.cuni.cz/app/user/953e620d-1bf0-11e8-9de3-00505601122b

import datetime
import logging
import os.path

import numpy as np
import tensorflow as tf

import lunar_lander_evaluator
import rl

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=None, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
    parser.add_argument("--alpha", default=0.5, type=float, help="Learning rate.")
    parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--input", "-i", default="lunar_lander_q_default.py")
    parser.add_argument("--output", "-o")
    parser.add_argument("--expert_trajectories", default=0, type=int,
                        help="Number of expert trajectories to learn from.")
    parser.add_argument("--steps", default=1, type=int, help="Number of steps for n-step learning.")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        run_name = args.run_name
    log_dir = os.path.join(args.log_dir, run_name)

    q = None
    if args.input is not None:
        try:
            q = rl.load(args.input)
        except FileNotFoundError:
            logging.info(f'Input model "{args.input}" not found.')
            args.train = True

    learner = rl.Learner(lunar_lander_evaluator.environment(), q=q, epsilon=args.epsilon, alpha=args.alpha,
                         gamma=args.gamma, steps=args.steps, render_each=args.render_each, log_dir=log_dir)

    if args.train:
        logging.info('Beginning learning from expert trajectories.')
        try:
            learner.learn_from_trajectories(args.expert_trajectories)
        finally:
            if args.output is not None:
                rl.save(args.output, learner.q)

        logging.info('Beginning training.')
        try:
            learner.perform(train=True, evaluate=False, episodes=args.episodes)
        finally:
            if args.output is not None:
                rl.save(args.output, learner.q)

    logging.info('Beginning evaluation.')
    learner.epsilon = 0.0
    learner.perform(train=False, evaluate=True, episodes=100)
