#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

import cart_pole_evaluator


class Network:
    def __init__(self, env, args):
        # TODO: Define suitable model. Apart from the model defined in `reinforce`,
        # define also another model `baseline`, which produces one output
        # (using a dense layer without activation).
        #
        # Use Adam optimizer with given `args.learning_rate` for both models.
        raise NotImplementedError()

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns,
                                                                                                       np.float32)

        # TODO: Train the model using the states, actions and observed returns.
        # You should:
        # - compute the predicted baseline using the `baseline` model
        # - train the policy model, using `returns - predicted_baseline` as weights
        #   in the sparse crossentropy loss
        # - train the `baseline` model to predict `returns`
        raise NotImplementedError()

    def predict(self, states):
        states = np.array(states, np.float32)
        # TODO: Predict distribution over actions for the given input states. Return
        # only the probabilities, not the baseline.
        raise NotImplementedError()


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=None, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=None, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=None, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=None, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=None, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # TODO(reinforce): Compute action probabilities using
                # `network.predict` and current `state`

                # TODO(reinforce): Choose `action` according to `probabilities` distribution (np.random.choice can be used)

                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO(reinforce): Compute returns by summing rewards (with discounting)

            # TODO(reinforce): Add states, actions and returns to the training batch

        # Train using the generated batch
        network.train(batch_states, batch_actions, batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            # TODO(reinforce): Compute action `probabilities` using `network.predict` and current `state`

            # Choose greedy action this time
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
