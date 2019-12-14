#!/usr/bin/env python3

import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

import gym_evaluator


class Network:
    def __init__(self, env, args, filepath=None):
        # Similarly to reinforce, define two models:
        # - _policy, which predicts distribution over the actions
        # - _value, which predicts the value function
        # Use independent networks for both of them, each with
        # `args.hidden_layer` neurons in one hidden layer,
        # and train them using Adam with given `args.learning_rate`.
        try:
            self._policy = load_model(f'{filepath}_policy.h5')
            self._value = load_model(f'{filepath}_value.h5')
        except OSError:
            self._policy = self.base_model(env, args)
            self._policy.add(Dense(env.actions, activation='softmax'))
            self._policy.compile(optimizer=Adam(lr=args.learning_rate), loss='sparse_categorical_crossentropy',
                                 metrics=['accuracy'], experimental_run_tf_function=False)
            self._value = self.base_model(env, args)
            self._value.add(Dense(1))
            self._value.compile(optimizer=Adam(lr=args.learning_rate), loss='mean_squared_error',
                                metrics=['accuracy'], experimental_run_tf_function=False)

    @staticmethod
    def base_model(env, args):
        model = Sequential()
        model.add(Dense(args.hidden_layer, input_shape=env.state_shape, activation='relu'))
        model.add(Dense(args.hidden_layer, activation='relu'))
        return model

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns,
                                                                                                       np.float32)
        # Train the policy network using policy gradient theorem
        # and the value network using MSE.
        baseline = self._value.predict_on_batch(states)
        returns_normalized = returns - baseline[:, 0]
        self._policy.train_on_batch(states, actions, sample_weight=returns_normalized)
        self._value.train_on_batch(states, returns)
        tf.summary.histogram('training_baseline', baseline)
        tf.summary.histogram('training_returns_normalized', returns_normalized)

    def predict_actions(self, states):
        states = np.array(states, np.float32)
        return self._policy.predict_on_batch(states)

    def predict_values(self, states):
        states = np.array(states, np.float32)
        return self._value.predict_on_batch(states)[:, 0]

    def save(self, filepath):
        self._policy.save(f'{filepath}_policy.h5')
        self._value.save(f'{filepath}_value.h5')


def evaluate(env, network, episodes=100, render_each=None, final_evaluation=False):
    returns = []
    for _ in range(episodes):
        returns.append(0)
        state, done = env.reset(final_evaluation), False
        while not done:
            if render_each and env.episode > 0 and env.episode % render_each == 0:
                env.render()
            probabilities = network.predict_actions([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
            returns[-1] += reward
    return returns


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
    parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--gamma", default=0.9, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=32, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--workers", default=128, type=int, help="Number of parallel workers.")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.summary.experimental.set_step(0)

    # Create the environment
    env = gym_evaluator.GymEnvironment(args.env)

    # Construct the network
    network = Network(env, args)

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('logs', run_name)
    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
        tf.summary.text('args', str(args))

    step = 0

    # Initialize parallel workers by env.parallel_init
    try:
        states = env.parallel_init(args.workers)
        while True:
            # Training
            for _ in range(args.evaluate_each):
                tf.summary.experimental.set_step(step)

                # Choose actions using network.predict_actions
                probabilities = network.predict_actions(states)
                assert probabilities.shape == (len(states), env.actions)
                assert np.allclose(probabilities.sum(axis=1), 1)
                actions = [np.random.choice(env.actions, p=p) for p in probabilities]
                assert len(actions) == len(states)

                # Perform steps by env.parallel_step
                steps = env.parallel_step(actions)

                # Compute return estimates by
                # - extracting next_states from steps
                # - computing value function approximation in next_states
                # - estimating returns by reward + (0 if done else args.gamma * next_state_value)
                next_states, rewards, dones, _ = zip(*steps)
                next_state_values = network.predict_values(next_states)
                tail_returns = next_state_values * args.gamma * np.logical_not(dones)
                returns = np.asarray(rewards) + tail_returns

                # Train network using current states, chosen actions and estimated returns
                with writer.as_default():
                    tf.summary.histogram('training_returns', returns)
                    network.train(states, actions, returns)

                states = next_states
                step += 1
            # Periodic evaluation
            returns = evaluate(gym_evaluator.GymEnvironment(args.env), network, episodes=args.evaluate_for, render_each=args.render_each)
            print("Step {}: Evaluation of {} episodes: {:.2f} +-{:.2f}".format(step, args.evaluate_for, np.mean(returns), np.std(returns)))
            with writer.as_default():
                tf.summary.histogram('evaluation_returns', returns)
                tf.summary.scalar('evaluation_returns.mean', np.mean(returns))
                tf.summary.scalar('evaluation_returns.std', np.std(returns))
            if np.mean(returns) > 475 and np.std(returns) < 25:
                break
    finally:
        network.save(f'paac_{run_name}')

    # On the end perform final evaluations with `env.reset(True)`
    logging.info('Final evaluation...')
    evaluate(gym_evaluator.GymEnvironment(args.env), network, render_each=args.render_each, final_evaluation=True)
