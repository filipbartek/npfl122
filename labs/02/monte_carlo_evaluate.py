#!/usr/bin/env python3
import cart_pole_evaluator
import numpy as np


def generate_episode(env, pi, epsilon=0.0, render=False):
    state, done = env.reset(True), False
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


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="monte_carlo_pi.py", help="Model Python file")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment()

    with open(args.model) as pi_file:
        from numpy import array
        pi = eval(pi_file.read())

    # Perform last 100 evaluation episodes
    # Stop exploring during evaluation.
    for evaluation_episode_i in range(100):
        generate_episode(env, pi, 0.0, args.render_each and env.episode and env.episode % args.render_each == 0)
