#!/usr/bin/env python3

import logging
import os
import time
from datetime import datetime
from itertools import count

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

import continuous_mountain_car_evaluator


# This class is a bare version of tfp.distributions.Normal
class Normal:
    def __init__(self, loc, scale):
        self.loc = tf.convert_to_tensor(loc, dtype=tf.float32)
        self.scale = tf.convert_to_tensor(scale, dtype=tf.float32)

    def log_prob(self, x):
        log_unnormalized = -0.5 * tf.math.squared_difference(x / self.scale, self.loc / self.scale)
        log_normalization = 0.5 * np.log(2. * np.pi) + tf.math.log(self.scale)
        return log_unnormalized - log_normalization

    def entropy(self):
        log_normalization = 0.5 * np.log(2. * np.pi) + tf.math.log(self.scale)
        entropy = 0.5 + log_normalization
        return entropy * tf.ones_like(self.loc)

    def sample_n(self, n, seed=None):
        shape = tf.concat([[n], tf.broadcast_dynamic_shape(tf.shape(self.loc), tf.shape(self.scale))], axis=0)
        sampled = tf.random.normal(shape=shape, mean=0., stddev=1., dtype=tf.float32, seed=seed)
        return sampled * self.scale + self.loc


class Network:
    def __init__(self, env, args):
        assert len(env.action_shape) == 1
        action_components = env.action_shape[0]

        self.entropy_regularization = args.entropy_regularization

        # TODO: Create `_model`, which: processes `states`. Because `states` are
        # vectors of tile indices, you need to convert them to one-hot-like
        # encoding. I.e., for batch example i, state should be a vector of
        # length `weights` with `tiles` ones on indices `states[i,
        # 0..`tiles`-1] and the rest being zeros.
        #
        # The model computes `mus` and `sds`, each of shape [batch_size, action_components].
        # Compute each independently using `states` as input, adding a fully connected
        # layer with args.hidden_layer units and ReLU activation. Then:
        # - For `mus` add a fully connected layer with `actions` outputs.
        #   To avoid `mus` moving from the required [-1,1] range, you can apply
        #   `tf.tanh` activation.
        # - For `sds` add a fully connected layer with `actions` outputs
        #   and `tf.nn.softplus` activation.
        # The model also computes `values`, starting with `states` and
        # - add a fully connected layer of size args.hidden_layer and ReLU activation
        # - add a fully connected layer with 1 output and no activation

        inputs = Input(shape=env.weights)
        mus_hidden = Dense(args.hidden_layer, activation='relu')(inputs)
        mus = Dense(action_components, activation='tanh')(mus_hidden)
        sds_hidden = Dense(args.hidden_layer, activation='relu')(inputs)
        sds = Dense(action_components, activation='softplus')(sds_hidden)
        policy = Concatenate(name='policy')([mus, sds])
        values_hidden = Dense(args.hidden_layer, activation='relu')(inputs)
        values = Dense(1, name='value')(values_hidden)

        self._model = Model(inputs=inputs, outputs=[policy, values])
        self._optimizer = Adam(args.learning_rate)

        def policy_loss_function(entropy_regularization):
            @tf.function
            def policy_loss(target_y, predicted_y):
                # Compute `loss` as a sum of two losses:
                # - negative log probability of the `actions` in the `action_distribution`
                #   (using `log_prob` method). You need to sum the log probabilities
                #   of subactions for a single batch example (using `tf.reduce_sum` with `axis=1`).
                #   Then weight the resulting vector by `(returns - tf.stop_gradient(values))`
                #   and compute its mean.
                # - negative value of the distribution entropy (use `entropy` method of
                #   the `action_distribution`) weighted by `args.entropy_regularization`.
                mus = predicted_y[:, 0]
                sds = predicted_y[:, 1]
                action_distribution = Normal(mus, sds)
                actions = target_y[:, 0]
                returns = target_y[:, 1]
                # TODO: Generalize for composite actions.
                loss = -action_distribution.log_prob(
                    actions) * returns - action_distribution.entropy() * entropy_regularization
                return loss

            return policy_loss

        assert tf.executing_eagerly()
        self._model.compile(self._optimizer,
                            loss={'policy': policy_loss_function(args.entropy_regularization), 'value': 'mse'},
                            experimental_run_tf_function=True)

        self.weights = env.weights

    def n_hot(self, states):
        # TODO: Try to use `K.one_hot`: https://www.tensorflow.org/api_docs/python/tf/keras/backend/one_hot
        # https://fdalvi.github.io/blog/2018-04-07-keras-sequential-onehot/
        x = np.zeros((len(states), self.weights), dtype=np.bool)
        for i, state in enumerate(states):
            x[i, state] = True
        return x

    def train(self, states, actions, returns):
        returns = np.asarray(returns, dtype=np.float32)
        baseline = self.predict_values(states)
        tf.summary.histogram('Baseline', baseline)
        returns_normalized = returns - baseline
        tf.summary.histogram('Discounted return normalized', returns_normalized)
        assert len(actions) == len(returns_normalized)
        action_returns = np.concatenate((actions, np.expand_dims(returns_normalized, axis=1)), axis=1)
        assert tf.executing_eagerly()
        metrics_values = self._model.train_on_batch(self.n_hot(states), {'policy': action_returns, 'value': returns})
        for name, value in zip(self._model.metrics_names, metrics_values):
            tf.summary.scalar(name, value)

    @tf.function
    def _predict(self, states):
        return self._model(states, training=False)

    def predict_actions(self, states):
        policy, _ = self._predict(self.n_hot(states))
        mus = policy[:, 0]
        sds = policy[:, 1]
        return mus.numpy(), sds.numpy()

    def predict_values(self, states):
        _, values = self._predict(self.n_hot(states))
        return values.numpy()[:, 0]


def evaluate(env, network, episodes=100, render_each=None, final_evaluation=False):
    returns = []
    for _ in range(episodes):
        returns.append(0)
        state, done = env.reset(final_evaluation), False
        while not done:
            if render_each and env.episode % render_each == 0:
                env.render()
            # Choose the mean action value.
            action = network.predict_actions([state])[0]
            assert all(env.action_ranges[0] <= action_component <= env.action_ranges[1] for action_component in action)
            state, reward, done, _ = env.step(action)
            returns[-1] += reward
    return returns


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
    parser.add_argument("--evaluate_each", default=1000, type=int, help="Evaluate each number of batches.")
    parser.add_argument("--evaluate_for", default=1, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=32, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--tiles", default=8, type=int, help="Tiles to use.")
    parser.add_argument("--workers", default=32, type=int, help="Number of parallel workers.")
    parser.add_argument("--batch_size", default=10, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    np.seterr(all='raise')

    logging.info('Initializing...')

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    assert tf.executing_eagerly()


    # Create the environment
    def new_environment(tiles):
        return continuous_mountain_car_evaluator.environment(tiles=tiles)


    env = new_environment(args.tiles)
    action_lows, action_highs = env.action_ranges

    # Construct the network
    network = Network(env, args)

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('logs', 'paac_continuous', run_name)
    writer_train = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    with writer_train.as_default():
        tf.summary.text('args', str(args), step=0)
    writer_evaluate = tf.summary.create_file_writer(os.path.join(log_dir, 'evaluate'))

    # Initialize parallel workers by env.parallel_init
    states = env.parallel_init(args.workers)
    start_step = np.zeros(args.workers, dtype=np.uint)
    last_return = np.full(args.workers, np.nan, dtype=np.float)
    worker_reward_sum = np.zeros(args.workers, dtype=np.float)
    time_start = time.time()
    batch_states = []
    batch_actions = []
    batch_returns = []
    try:
        logging.info('Training...')
        for step in tqdm(count(), unit='step'):
            tf.summary.experimental.set_step(step)
            # Training
            # Choose actions using network.predict_actions.
            # using np.random.normal to sample action and np.clip
            # to clip it using action_lows and action_highs,
            mus, sds = network.predict_actions(states)
            actions = np.clip(np.random.normal(mus, sds), action_lows, action_highs)
            assert actions.shape == (args.workers,)
            actions = np.expand_dims(actions, axis=1)

            # Perform steps by env.parallel_step
            assert actions.shape == tuple([args.workers] + env.action_shape)
            assert np.all(actions >= action_lows)
            assert np.all(actions <= action_highs)
            steps = env.parallel_step(actions)

            # Compute return estimates by
            # - extracting next_states from steps
            # - computing value function approximation in next_states
            # - estimating returns by reward + (0 if done else args.gamma * next_state_value)
            next_states, rewards, dones, _ = zip(*steps)
            next_state_values = network.predict_values(next_states)
            tail_returns = next_state_values * args.gamma * np.logical_not(dones)
            returns = np.asarray(rewards) + tail_returns

            for i in range(args.workers):
                worker_reward_sum[i] += rewards[i]
                if dones[i]:
                    last_return[i] = worker_reward_sum[i]
                    start_step[i] = step
                    worker_reward_sum[i] = 0

            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)

            with writer_train.as_default():
                if step % 100 == 0:
                    tf.summary.histogram('Discounted return', returns)
                    if not np.isnan(last_return).any():
                        tf.summary.histogram('Return', last_return)
                        tf.summary.scalar('Return mean', last_return.mean())
                if len(batch_states) >= args.workers * args.batch_size:
                    # Train network using current states, chosen actions and estimated returns
                    network.train(batch_states, batch_actions, batch_returns)
                    batch_states = []
                    batch_actions = []
                    batch_returns = []

            if args.evaluate_each and step % args.evaluate_each == 0:
                logging.debug(f'Step {step}: training: {time.time() - time_start} s')
                time_start = time.time()
                # Periodic evaluation
                returns = evaluate(new_environment(args.tiles), network, episodes=args.evaluate_for,
                                   render_each=args.render_each)
                with writer_evaluate.as_default():
                    tf.summary.histogram('Return', returns)
                    tf.summary.scalar('Return mean', np.mean(returns))
                logging.debug(f'Step {step}: evaluation: {time.time() - time_start} s')
                time_start = time.time()
    finally:
        pass

    # On the end perform final evaluations with `env.reset(True)`
    logging.info('Final evaluation...')
    evaluate(new_environment(args.tiles), network, render_each=args.render_each, final_evaluation=True)
