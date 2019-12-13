#!/usr/bin/env python3

import logging
import os
from collections import deque
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

import cart_pole_pixels_evaluator


class Network:
    def __init__(self, env, args, name='cart_pole_pixels'):
        self.loaded = False
        try:
            self.pi = load_model(f'{name}_pi.h5')
            self.pi.optimizer.lr = args.learning_rate
            self.v = load_model(f'{name}_v.h5')
            self.v.optimizer.lr = args.learning_rate
            self.loaded = True
            logging.info('Model loaded.')
        except OSError:
            logging.info('Model file not found. Creating a new model.')
            self.pi = self.base_model()
            self.pi.add(Dense(env.actions, activation='softmax'))
            self.pi.compile(optimizer=Adam(lr=args.learning_rate), loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'], experimental_run_tf_function=False)
            self.v = self.base_model()
            self.v.add(Dense(1))
            self.v.compile(optimizer=Adam(lr=args.learning_rate), loss='mean_squared_error',
                           metrics=['accuracy'], experimental_run_tf_function=False)

    def base_model(self):
        # https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        model = Sequential()
        model.add(Conv2D(16, (8, 8), (4, 4), activation='relu', input_shape=env.state_shape))
        model.add(Conv2D(32, (4, 4), (2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        return model

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns,
                                                                                                       np.float32)
        tf.summary.histogram('actions', actions)
        tf.summary.histogram('returns', returns)
        baseline = self.v.predict_on_batch(states)
        assert baseline.shape == (len(returns), 1)
        tf.summary.histogram('baseline', baseline)
        returns_normalized = returns - baseline[:, 0]
        tf.summary.histogram('returns_normalized', returns_normalized)
        tf.summary.scalar('returns.mean', returns.mean())
        metrics_values = self.pi.train_on_batch(states, actions, sample_weight=returns_normalized)
        for name, value in logs(self.pi.metrics_names, metrics_values).items():
            tf.summary.scalar(f'pi.{name}', value)
        metrics_values = self.v.train_on_batch(states, returns)
        for name, value in logs(self.v.metrics_names, metrics_values).items():
            tf.summary.scalar(f'v.{name}', value)

    def predict(self, states):
        return self.pi.predict_on_batch(np.array(states, np.float32))


def logs(metrics_names, metrics_values):
    if type(metrics_values) is not list:
        metrics_values = [metrics_values]
    return {name: value for name, value in zip(metrics_names, metrics_values)}


def validate(env, network, episodes=100, evaluation=False):
    returns = []
    for _ in range(episodes):
        state, done = env.reset(evaluation), False
        episode_return = 0
        while not done:
            probabilities = network.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
            episode_return += reward
        returns.append(episode_return)
    return returns


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=0.9, type=float, help="Discounting factor.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
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

    logging.info(tf.config.experimental.list_physical_devices('GPU'))

    def new_environment():
        return cart_pole_pixels_evaluator.environment()

    try:
        import cart_pole_pixels_model
    except ModuleNotFoundError:
        pass

    task_name = 'cart_pole_pixels'
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('logs', run_name)
    writer_train = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))

    env = new_environment()
    network = Network(env, args, name=task_name)

    if not network.loaded or args.retrain:
        returns = deque(maxlen=100)
        best_returns_mean = None
        logging.info('Training...')
        try:
            for batch_i in range(args.episodes // args.batch_size):
                batch_states, batch_actions, batch_returns = [], [], []
                batch_return = 0
                for batch_episode_i in range(args.batch_size):
                    tf.summary.experimental.set_step(batch_i * args.batch_size + batch_episode_i)
                    states, actions, rewards = [], [], []
                    state, done = env.reset(), False
                    while not done:
                        if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                            env.render()
                        probabilities = network.predict([state])[0]
                        assert len(probabilities) == env.actions
                        # Normalize to compensate for instabilities
                        probabilities = np.asarray(probabilities) / np.sum(probabilities)
                        action = np.random.choice(env.actions, p=probabilities)
                        next_state, reward, done, _ = env.step(action)
                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        state = next_state
                    cur_return = np.sum(rewards)
                    returns.append(cur_return)
                    mean_return = np.mean(returns)
                    if len(returns) >= 100 and (best_returns_mean is None or mean_return > best_returns_mean):
                        logging.info(f'New best: {mean_return}')
                        best_returns_mean = mean_return
                        network.pi.save(f'{task_name}_{run_name}_best_pi.h5')
                        network.v.save(f'{task_name}_{run_name}_best_v.h5')
                    with writer_train.as_default():
                        tf.summary.scalar('return', cur_return)
                        tf.summary.scalar('return.mean', mean_return)
                    batch_states.extend(states)
                    batch_actions.extend(actions)
                    batch_returns.extend(
                        np.sum((rewards[t] * np.power(args.gamma, t - start) for t in range(start, len(rewards)))) for
                        start in range(len(rewards)))
                    assert len(batch_states) == len(batch_actions) == len(batch_returns)
                with writer_train.as_default():
                    network.train(batch_states, batch_actions, batch_returns)
        finally:
            network.pi.save(f'{task_name}_{run_name}_pi.h5')
            network.v.save(f'{task_name}_{run_name}_v.h5')

    # Final evaluation
    logging.info('Evaluating...')
    validate(new_environment(), network, episodes=100, evaluation=True)
