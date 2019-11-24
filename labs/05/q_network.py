#!/usr/bin/env python3

import collections
import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import RMSprop

import cart_pole_evaluator


class Network:
    def __init__(self, env, args, summary_writer=None, filepath=None):
        self.state_shape = env.state_shape
        self.actions = env.actions
        self.gamma = args.gamma
        self.summary_writer = summary_writer

        try:
            self.model = load_model(filepath)
        except OSError:
            # TODO: Add regularizers (?).
            # TODO: Initialize more reasonably. The q values should be e.g. uniformly between 0 and 500.
            self.model = Sequential()
            self.model.add(Dense(args.hidden_layer_size, input_shape=env.state_shape, activation='relu'))
            for _ in range(args.hidden_layers):
                self.model.add(Dense(args.hidden_layer_size, activation='relu'))
            self.model.add(Dense(env.actions))
            self.model.compile(optimizer=RMSprop(lr=args.learning_rate), loss='mse')

        self.model_target = Sequential.from_config(self.model.get_config())
        self.update_target()

    def update_target(self):
        self.model_target.set_weights(self.model.get_weights())

    def train(self, transitions, step=None):
        x, y = self.get_batch(transitions)
        scalars = self.model.train_on_batch(x, y)
        if step % 100 == 0:
            self.update_target()
        if type(scalars) is np.float32:
            scalars = [scalars]
        if self.summary_writer is not None:
            with self.summary_writer.as_default():
                for name, value in zip(self.model.metrics_names, scalars):
                    tf.summary.scalar(name, value, step=step)

    def get_batch(self, transitions):
        x = np.empty([len(transitions)] + self.state_shape)
        y = np.empty((len(transitions), self.actions))
        q_this = self.predict(np.asarray([transition.state for transition in transitions]))
        q_next = self.predict(np.asarray([transition.next_state for transition in transitions]), target=True)
        for i, transition in enumerate(transitions):
            x[i] = transition.state
            cur_q = q_this[i]
            cur_q[transition.action] = transition.reward
            if not transition.done:
                cur_q[transition.action] += self.gamma * q_next[i].max()
            y[i] = cur_q
        return x, y

    def predict(self, states, target=False):
        if not target:
            return self.model.predict_on_batch(states)
        else:
            return self.model_target.predict_on_batch(states)

    def validate(self, env, episodes=100, evaluate=False, render_each=None):
        episode_rewards = np.zeros(episodes, dtype=np.float)
        for episode in range(episodes):
            state, done = env.reset(evaluate), False
            while not done:
                if render_each and env.episode % render_each == 0:
                    env.render()
                action = np.argmax(self.predict(np.array([state], np.float32))[0])
                state, reward, done, _ = env.step(action)
                episode_rewards[episode] += reward
        return episode_rewards.mean()


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--episodes", default=0, type=int, help="Episodes for epsilon decay.")
    parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=1, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=20, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--input", "-i", default="q_network_model.h5")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    summary_writer_train = tf.summary.create_file_writer(os.path.join('logs', run_name, 'train'))
    summary_writer_validate = tf.summary.create_file_writer(os.path.join('logs', run_name, 'validate'))

    # Construct the network
    try:
        import q_network_model
        assert os.path.isfile("q_network_model.h5")
    except ModuleNotFoundError:
        pass

    network = Network(env, args, summary_writer_train, args.input)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    evaluating = False
    epsilon = args.epsilon
    training = True
    step = 0
    try:
        while env.episode < args.episodes:
            # Perform episode
            state, done = env.reset(), False
            while not done:
                if env.episode % 100 == 0:
                    validation_reward = network.validate(cart_pole_evaluator.environment(discrete=False),
                                                         render_each=args.render_each)
                    with summary_writer_validate.as_default():
                        tf.summary.scalar('reward', validation_reward, step=env.episode)

                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                if epsilon > 0.0 and np.random.random_sample() < epsilon:
                    action = np.random.randint(env.actions)
                else:
                    q_values = network.predict(np.array([state], np.float32))[0]
                    action = q_values.argmax()

                next_state, reward, done, _ = env.step(action)

                # Append state, action, reward, done and next_state to replay_buffer
                replay_buffer.append(Transition(state, action, reward, done, next_state))

                if len(replay_buffer) >= args.batch_size * 8:
                    indexes = np.random.choice(len(replay_buffer), args.batch_size)
                    transitions = [replay_buffer[i] for i in indexes]
                    network.train(transitions, step=step)

                state = next_state
                step += 1

            if args.epsilon_final:
                epsilon = np.exp(
                    np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
    finally:
        network.model.save(f'q_network_{run_name}.h5')

    # Final evaluation
    network.validate(cart_pole_evaluator.environment(discrete=False), evaluate=True, render_each=args.render_each)
