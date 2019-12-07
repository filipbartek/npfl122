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
            self.model = load_model('reinforce_model.h5')
            self.loaded = True
            logging.info('Model loaded from reinforce_mode.h5.')
        except OSError:
            logging.info('Model file not found. Creating a new model.')
            self.model = Sequential()
            self.model.add(Dense(args.hidden_layer_size, input_shape=env.state_shape, activation='relu'))
            for _ in range(args.hidden_layers):
                self.model.add(Dense(args.hidden_layer_size, activation='relu'))
            self.model.add(Dense(env.actions, activation='softmax'))
            self.model.compile(
                optimizer=Adam(lr=args.learning_rate, decay=args.learning_rate * args.batch_size / args.episodes),
                loss='sparse_categorical_crossentropy', metrics=['accuracy'], experimental_run_tf_function=False)

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns,
                                                                                                       np.float32)
        # Train the model using the states, actions and observed returns.
        # Use `returns` as weights in the sparse crossentropy loss.
        metrics_values = self.model.train_on_batch(states, actions, sample_weight=returns)
        for name, value in self.logs(metrics_values).items():
            tf.summary.scalar(name, value)

    def logs(self, metrics_values):
        if type(metrics_values) is not list:
            metrics_values = [metrics_values]
        return {name: value for name, value in zip(self.model.metrics_names, list(metrics_values))}

    def predict(self, states):
        states = np.array(states, np.float32)
        # Predict distribution over actions for the given input states.
        return self.model.predict_on_batch(states)

    def validate(self, env, episodes, evaluation=False):
        returns = []
        for _ in range(episodes):
            state, done = env.reset(evaluation), False
            rewards = []
            while not done:
                # Compute action `probabilities` using `network.predict` and current `state`
                probabilities = self.predict([state])[0]

                # Choose greedy action this time
                action = np.argmax(probabilities)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
            returns.append(np.sum(rewards))
        # We cannot alias the summary names.
        tf.summary.histogram('returns', returns)
        tf.summary.scalar('return', np.mean(returns))
        tf.summary.scalar('return.std', np.std(returns))
        return returns


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=0.9, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=1, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=20, type=int, help="Size of hidden layer.")
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

    # Construct the network
    try:
        import reinforce_model

        assert os.path.isfile("reinforce_model.h5")
    except ModuleNotFoundError:
        pass

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(env, args)

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join('logs', run_name)
    writer_train = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
    writer_validate_10 = tf.summary.create_file_writer(os.path.join(logdir, 'validate_10'))
    writer_validate_100 = tf.summary.create_file_writer(os.path.join(logdir, 'validate_100'))

    if not network.loaded or args.retrain:
        logging.info('Training...')
        # Training
        try:
            for batch_i in range(args.episodes // args.batch_size):
                batch_states, batch_actions, batch_returns = [], [], []
                for batch_episode_i in range(args.batch_size):
                    tf.summary.experimental.set_step(batch_i * args.batch_size + batch_episode_i)

                    # Perform episode
                    states, actions, rewards = [], [], []
                    state, done = env.reset(), False
                    while not done:
                        if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                            env.render()

                        # Compute action probabilities using `network.predict` and current `state`
                        probabilities = network.predict([state])[0]

                        # Choose `action` according to `probabilities` distribution (np.random.choice can be used)
                        assert len(probabilities) == env.actions
                        action = np.random.choice(env.actions, p=probabilities)

                        next_state, reward, done, _ = env.step(action)

                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)

                        state = next_state

                    with writer_train.as_default():
                        tf.summary.scalar('return', np.sum(rewards))

                    # Compute returns by summing rewards (with discounting)
                    # Add states, actions and returns to the training batch
                    batch_states.extend(states)
                    batch_actions.extend(actions)
                    batch_returns.extend(
                        np.sum((rewards[t] * np.power(args.gamma, t - start) for t in range(start, len(rewards)))) for start
                        in
                        range(len(rewards)))
                    assert len(batch_states) == len(batch_actions) == len(batch_returns)

                # Train using the generated batch
                with writer_train.as_default():
                    network.train(batch_states, batch_actions, batch_returns)

                if batch_i % 10 == 0:
                    with writer_validate_10.as_default():
                        returns_10 = network.validate(cart_pole_evaluator.environment(discrete=False), 10)
                    if np.mean(returns_10) >= 490:
                        with writer_validate_100.as_default():
                            returns_100 = network.validate(cart_pole_evaluator.environment(discrete=False), 100)
                        if np.mean(returns_100) >= 495 and np.std(returns_100) <= 50:
                            logging.info('Victory!')
                            break
            with writer_validate_10.as_default():
                returns_10 = network.validate(cart_pole_evaluator.environment(discrete=False), 10)
            with writer_validate_100.as_default():
                returns_100 = network.validate(cart_pole_evaluator.environment(discrete=False), 100)
        finally:
            network.model.save(f'reinforce_{run_name}.h5')

    # Final evaluation
    logging.info('Evaluating...')
    network.validate(cart_pole_evaluator.environment(discrete=False), 100, True)
