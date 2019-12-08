#!/usr/bin/env python3

import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

import cart_pole_evaluator


class Network:
    def __init__(self, env, args):
        self.loaded = False
        try:
            self.pi = load_model('reinforce_baseline_pi.h5')
            self.v = load_model('reinforce_baseline_v.h5')
            self.loaded = True
            logging.info('Model loaded.')
        except OSError:
            logging.info('Model file not found. Creating a new model.')
            self.pi = Sequential()
            self.pi.add(Dense(args.hidden_layer_size, input_shape=env.state_shape, activation='relu'))
            for _ in range(args.hidden_layers):
                self.pi.add(Dense(args.hidden_layer_size, activation='relu'))
            self.pi.add(Dense(env.actions, activation='softmax'))
            self.pi.compile(optimizer=Adam(lr=args.learning_rate), loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'], experimental_run_tf_function=False)
            self.v = Sequential()
            self.v.add(Dense(args.hidden_layer_size, input_shape=env.state_shape, activation='relu'))
            for _ in range(args.hidden_layers):
                self.v.add(Dense(args.hidden_layer_size, activation='relu'))
            self.v.add(Dense(1))
            self.v.compile(optimizer=Adam(lr=args.learning_rate), loss='mean_squared_error',
                           metrics=['accuracy'], experimental_run_tf_function=False)

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns,
                                                                                                       np.float32)
        baseline = self.v.predict_on_batch(states)
        self.pi.train_on_batch(states, actions, sample_weight=returns.flatten() - baseline.flatten())
        self.v.train_on_batch(states, returns)
        # TODO: Publish stats in TensorBoard.

    def predict(self, states):
        states = np.array(states, np.float32)
        # Predict distribution over actions for the given input states. Return
        # only the probabilities, not the baseline.
        return self.pi.predict_on_batch(states)


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=0.9, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=1, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=300, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    try:
        import reinforce_baseline_model

        assert os.path.isfile("reinforce_baseline_pi.h5")
        assert os.path.isfile("reinforce_baseline_v.h5")
    except ModuleNotFoundError:
        pass

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(env, args)

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    if not network.loaded or args.retrain:
        logging.info('Training...')
        try:
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

                        # Compute action probabilities using
                        # `network.predict` and current `state`
                        probabilities = network.predict([state])[0]

                        # Choose `action` according to `probabilities` distribution (np.random.choice can be used)
                        assert len(probabilities) == env.actions
                        action = np.random.choice(env.actions, p=probabilities)

                        next_state, reward, done, _ = env.step(action)

                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)

                        state = next_state

                    # Compute returns by summing rewards (with discounting)
                    # Add states, actions and returns to the training batch
                    batch_states.extend(states)
                    batch_actions.extend(actions)
                    batch_returns.extend(
                        np.sum((rewards[t] * np.power(args.gamma, t - start) for t in range(start, len(rewards)))) for start
                        in range(len(rewards)))
                    assert len(batch_states) == len(batch_actions) == len(batch_returns)

                # Train using the generated batch
                network.train(batch_states, batch_actions, batch_returns)
        finally:
            network.pi.save(f'reinforce_baseline_{run_name}_pi.h5')
            network.v.save(f'reinforce_baseline_{run_name}_v.h5')

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            # Compute action `probabilities` using `network.predict` and current `state`
            probabilities = network.predict([state])[0]

            # Choose greedy action this time
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
