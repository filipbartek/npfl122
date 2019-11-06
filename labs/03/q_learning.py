#!/usr/bin/env python3

import logging
import sys

import numpy as np
import scipy.stats

import mountain_car_evaluator

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="q_learning_model.py")
    parser.add_argument("--output", "-o")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--episodes", default=None, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
    parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
    parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--target_reward", default=-140, type=int,
                        help="Terminate training as soon as the 100-episode mean reward reaches this value.")
    parser.add_argument("--live_plot_period", type=int,
                        help="Plot training statistics once per this number of episodes.")
    parser.add_argument("--evaluation_episodes", default=100, type=int)
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment()

    if args.live_plot_period is not None:
        try:
            from livelossplot import PlotLosses

            liveplot = PlotLosses()
        except ModuleNotFoundError:
            pass

    # See npfl122-03.pdf:19
    epsilon = args.epsilon

    q = np.zeros((env.states, env.actions), dtype=np.float)
    assert args.input is not None
    try:
        with open(args.input) as model_file:
            from numpy import array, uint64

            model = eval(model_file.read())
            q = model['q']
            assert q.shape == (env.states, env.actions)
            logging.info(f'Model loaded from "{args.input}".')
    except FileNotFoundError:
        logging.info(f'Could not load model from "{args.input}".')
        args.train = True

    try:
        if args.train:
            alpha = args.alpha
            gamma = args.gamma
            returns = []
            window_size = 100
            slope = 0
            training = True
            logging.info('Beginning training.')
            # TODO: Parallelize.
            while training:
                # Perform a training episode
                state, done = env.reset(), False
                episode_reward = 0
                while not done:
                    if args.render_each and env.episode and env.episode % args.render_each == 0:
                        env.render()
                    # Choose next action using epsilon-greedy policy.
                    if np.random.random_sample() < epsilon:
                        action = np.random.randint(env.actions)
                    else:
                        action = np.argmax(q[state, :])
                    next_state, reward, done, _ = env.step(action)
                    q[state, action] += alpha * (reward + gamma * np.max(q[next_state, :]) - q[state, action])
                    state = next_state
                    episode_reward += reward
                returns.append(episode_reward)
                if len(returns) >= window_size / 2:
                    slope, _, _, _, _ = scipy.stats.linregress(np.arange(min(window_size, len(returns))),
                                                               returns[-window_size:])
                    if len(returns) >= window_size and slope < 0.01:
                        epsilon *= 0.95
                window_mean = np.mean(returns[-window_size:])
                window_std = np.std(returns[-window_size:])
                # Break on convergence (score significantly higher than `args.target_reward`).
                if len(returns) >= window_size and window_mean >= args.target_reward:
                    logging.info('Converged.')
                    break
                if 'liveplot' in locals():
                    log = {'episode reward': episode_reward,
                           'epsilon': epsilon,
                           'episode return mean': window_mean,
                           'episode return std': window_std,
                           'episode return trend': slope
                           }
                    liveplot.update(log)
                    if env.episode and env.episode % args.live_plot_period == 0:
                        liveplot.draw()
    finally:
        if 'liveplot' in locals() and env.episode and env.episode >= 1:
            liveplot.draw()

        if args.output is not None:
            with open(args.output, 'w') as model_file:
                # https://stackoverflow.com/a/57891767/4054250
                options = np.get_printoptions()
                np.set_printoptions(threshold=sys.maxsize)
                model = {
                    'episode': env.episode,
                    'epsilon': epsilon,
                    'q': q
                }
                model_file.write(repr(model))
                np.set_printoptions(**options)
                logging.info(f'Model saved into "{args.output}".')

        logging.info('Beginning evaluation.')
        # Perform evaluation episodes
        for _ in range(args.evaluation_episodes):
            state, done = env.reset(True), False
            while not done:
                if args.render_each and env.episode and env.episode % args.render_each == 0:
                    env.render()
                action = np.argmax(q[state, :])
                state, reward, done, _ = env.step(action)
