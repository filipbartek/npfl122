#!/usr/bin/env python3
import gym
import numpy as np


def pi(action):
    result = 0
    if action in [1, 2]:
        result = 1 / 2
    return result


def b(actions):
    return 1 / actions


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    args = parser.parse_args()

    gamma = 1.0

    # Create the environment
    env = gym.make("FrozenLake-v0")
    env.seed(42)
    states = env.observation_space.n
    actions = env.action_space.n

    # Behaviour policy is uniformly random.
    # Target policy uniformly chooses either action 1 or 2.
    V = np.zeros(states)  # Value function of target policy (choosing 1 or 2 uniformly)
    C = np.zeros(states)

    try:
        import tqdm
        t = tqdm.tqdm(total=args.episodes, unit="episode")
    except ModuleNotFoundError:
        pass

    for _ in range(args.episodes):
        state, done = env.reset(), False

        # Generate episode
        episode = []
        while not done:
            action = np.random.choice(actions)  # Behaviour policy: choose any action uniformly.
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # Update V using weighted importance sampling.
        # See npfl122-03.pdf:30
        g = 0
        w = 1  # rho
        for state, action, reward in reversed(episode):
            w *= pi(action) / b(actions)
            if w == 0:
                break
            g = gamma * g + reward
            C[state] += w
            V[state] += w / C[state] * (g - V[state])

        if 't' in locals():
            t.set_postfix({"value of state 0": V[0]})
            t.update()

    if 't' in locals():
        t.close()

    # Print the final value function V
    for row in V.reshape(4, 4):
        print(" ".join(["{:5.2f}".format(x) for x in row]))
