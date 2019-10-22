#!/usr/bin/env python3
import argparse
import warnings

import numpy as np
import scipy.special

class MultiArmedBandits():
    def __init__(self, bandits, episode_length, seed=42):
        self._generator = np.random.RandomState(seed)

        self._bandits = []
        for _ in range(bandits):
            self._bandits.append(self._generator.normal(0., 1.))
        self._done = True
        self._episode_length = episode_length

    def reset(self):
        self._done = False
        self._trials = 0
        return None

    def step(self, action):
        if self._done:
            raise ValueError("Cannot step in MultiArmedBandits when there is no running episode")
        self._trials += 1
        self._done = self._trials == self._episode_length
        reward = self._generator.normal(self._bandits[action], 1.)
        return None, reward, self._done, {}

parser = argparse.ArgumentParser()
parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
parser.add_argument("--episodes", default=100, type=int, help="Training episodes.")
parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")

parser.add_argument("--mode", default="greedy", type=str, help="Mode to use -- greedy, ucb and gradient.")
parser.add_argument("--alpha", default=0, type=float, help="Learning rate to use (if applicable).")
parser.add_argument("--c", default=1, type=float, help="Confidence level in ucb (if applicable).")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor (if applicable).")
parser.add_argument("--initial", default=0, type=float, help="Initial value function levels (if applicable).")

def main(args):
    # Fix random seed
    np.random.seed(args.seed)

    # Create environment
    env = MultiArmedBandits(args.bandits, args.episode_length)

    if args.mode == "gradient" and args.alpha == 0:
        warnings.warn("alpha is zero.")

    episode_average_rewards = np.empty(args.episodes)

    try:
        import tqdm
        t = tqdm.tqdm(desc=args.mode, total=args.episodes, unit="episode")
    except ModuleNotFoundError:
        pass

    for episode in range(args.episodes):
        env.reset()

        # Initialize parameters (depending on mode).
        current_episode_trials = 0
        current_episode_total_reward = 0
        if args.mode in ["greedy", "ucb"]:
            n = np.zeros(args.bandits, dtype=np.int)
            q = np.full(args.bandits, args.initial, dtype=np.float)
        if args.mode == "gradient":
            h = np.zeros(args.bandits, dtype=np.float)

        done = False
        while not done:
            # Action selection according to mode
            action = None
            if args.mode == "greedy":
                if np.random.random() < args.epsilon:
                    # Explore
                    action = np.random.choice(args.bandits)
                else:
                    # Exploit
                    action = np.argmax(q)
            elif args.mode == "ucb":
                if current_episode_trials == 0:
                    assert np.all(n == 0)
                    assert np.all(q == q[0])
                    action = np.random.choice(args.bandits)
                else:
                    with np.errstate(divide='ignore'):
                        assert np.log(current_episode_trials + 1) != 0
                        action = np.argmax(q + args.c * np.sqrt(np.log(current_episode_trials + 1) / n))
            elif args.mode == "gradient":
                pi = scipy.special.softmax(h)
                action = np.random.choice(args.bandits, p=pi)

            _, reward, done, _ = env.step(action)

            # Update parameters
            current_episode_trials += 1
            current_episode_total_reward += reward
            if args.mode in ["greedy", "ucb"]:
                n[action] += 1
                if args.alpha == 0:
                    q[action] += (reward - q[action]) / n[action]
                else:
                    assert args.alpha > 0
                    assert args.alpha <= 1
                    q[action] += (reward - q[action]) * args.alpha
            if args.mode == "gradient":
                h += args.alpha * reward * (np.eye(args.bandits)[action] - pi)

        episode_average_rewards[episode] = current_episode_total_reward / current_episode_trials

        if 't' in locals():
            t.set_postfix({"mean": episode_average_rewards[:episode + 1].mean(), "std": episode_average_rewards[:episode + 1].std()})
            t.update()

    if 't' in locals():
        t.close()

    # For every episode, compute its average reward (a single number),
    # obtaining `args.episodes` values. Then return the final score as
    # mean and standard deviation of these `args.episodes` values.
    return episode_average_rewards.mean(), episode_average_rewards.std()

if __name__ == "__main__":
    mean, std = main(parser.parse_args())
    # Print the mean and std for ReCodEx to validate
    print("{:.2f} {:.2f}".format(mean, std))
