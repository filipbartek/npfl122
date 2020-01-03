#!/usr/bin/env python3

# Team members:
# Filip Bartek       | https://recodex.mff.cuni.cz/app/user/9d1ef2af-eb87-11e9-9ce9-00505601122b
# Bartosz Piotrowski | https://recodex.mff.cuni.cz/app/user/953e620d-1bf0-11e8-9de3-00505601122b
# Pavel Lucivnak     | https://recodex.mff.cuni.cz/app/user/8b0e9fd8-e9ae-11e9-9ce9-00505601122b

import logging
import os
from collections import namedtuple, deque
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, concatenate, Lambda
from tensorflow.keras.models import clone_model, Model
from tensorflow.keras.optimizers import Adam

import gym_evaluator

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


def update_model(source_model, target_model, update):
    for source, target in zip(source_model.variables, target_model.variables):
        assert source.shape == target.shape
        assert source.dtype == target.dtype
        update(source, target)


def copy_weights(source_model, target_model):
    update_model(source_model, target_model, lambda source, target: target.assign(source))


def clone_model_with_weights(source_model):
    target_model = clone_model(source_model)
    copy_weights(source_model, target_model)
    assert all(np.array_equal(source.numpy(), target.numpy()) for source, target in
               zip(source_model.variables, target_model.variables))
    return target_model


class Network:
    def __init__(self, env, args):
        self.gamma = args.gamma
        self.target_tau = args.target_tau
        self.loaded = False

        self._value = self._new_value_model(env, args.hidden_layer)
        # keras.utils.plot_model(self._value, 'walker_value.png', show_shapes=True)
        self._value.compile(optimizer=Adam(args.learning_rate), loss='mse', experimental_run_tf_function=False)
        self._target_value = clone_model_with_weights(self._value)

        self._value_twin = self._new_value_model(env, args.hidden_layer)
        self._value_twin.compile(optimizer=Adam(args.learning_rate), loss='mse', experimental_run_tf_function=False)
        self._target_value_twin = clone_model_with_weights(self._value_twin)

        self._policy = self._new_policy_model(env, args.hidden_layer)
        # keras.utils.plot_model(self._policy, 'walker_policy.png', show_shapes=True)
        self._target_policy = clone_model_with_weights(self._policy)

        self._policy_value = self._new_policy_value_model(env)
        # keras.utils.plot_model(self._policy_value, 'walker_policy_value.png', show_shapes=True)
        self._policy_value_optimizer = Adam(args.learning_rate)

        self.state_models = {
            'value': self._value,
            'target_value': self._target_value,
            'value_twin': self._value_twin,
            'target_value_twin': self._target_value_twin,
            'policy': self._policy,
            'target_policy': self._target_policy
        }

        self._checkpoint = tf.train.Checkpoint(**self.state_models, policy_value_optimizer=self._policy_value_optimizer)

    @staticmethod
    def _new_value_model(env, hidden_layer_size):
        input_state = Input(env.state_shape, name='state')
        input_action = Input(env.action_shape, name='action')
        x = input_state
        x = Dense(hidden_layer_size, activation='relu', name='state_relu')(x)
        x = concatenate([x, input_action])
        x = Dense(hidden_layer_size, activation='relu', name='relu_0')(x)
        x = Dense(hidden_layer_size, activation='relu', name='relu_1')(x)
        x = Dense(1, name='estimated_return')(x)
        output = x
        return Model(inputs=[input_state, input_action], outputs=output, name='value')

    @staticmethod
    def _new_policy_model(env, hidden_layer_size):
        input = Input(env.state_shape, name='state')
        x = input
        x = Dense(hidden_layer_size, activation='relu', name='relu_0')(x)
        x = Dense(hidden_layer_size, activation='relu', name='relu_1')(x)
        assert len(env.action_shape) == 1
        x = Dense(env.action_shape[0], activation='tanh', name='tanh')(x)
        action_ranges = np.asarray(env.action_ranges, dtype=np.float32)
        assert action_ranges.shape[0] == 2
        if np.any(action_ranges[0] != -1) or np.any(action_ranges[1] != 1):
            assert action_ranges.shape[1] >= 1
            scale = action_ranges[1, 0]
            if np.all(action_ranges[0] == -scale) and np.all(action_ranges[1] == scale):
                logging.debug(f'Rescaling actions by scalar multiplication. Scale: {scale}')

                def rescale(x):
                    return x * scale
            else:
                logging.debug('Rescaling actions by interpolation.')

                import tensorflow_probability as tfp

                def rescale(x):
                    return tfp.math.batch_interp_regular_1d_grid(x, -1, 1, action_ranges.transpose())
            x = Lambda(rescale, name='rescale')(x)
        else:
            logging.debug('Actions rescaling not necessary.')
        output = x
        return Model(inputs=input, outputs=output, name='policy')

    def _new_policy_value_model(self, env):
        state = Input(env.state_shape, name='state')
        return Model(inputs=state, outputs=self._value([state, self._policy(state)]), name='policy_value')

    @staticmethod
    def write_metrics(model_name, metrics_names, metrics_values):
        if not isinstance(metrics_values, list):
            metrics_values = [metrics_values]
        for name, value in zip(metrics_names, metrics_values):
            tf.summary.scalar(f'{model_name}.{name}', value)

    def tau_update(self, source_model, target_model):
        update_model(source_model, target_model,
                     lambda source, target: target.assign_add(self.target_tau * (source - target)))

    def train_value(self, states, actions, returns):
        metrics_values = self._value.train_on_batch({'state': states, 'action': actions}, returns)
        self.write_metrics('value', self._value.metrics_names, metrics_values)
        self.tau_update(self._value, self._target_value)

        metrics_values = self._value_twin.train_on_batch({'state': states, 'action': actions}, returns)
        self.write_metrics('value_twin', self._value_twin.metrics_names, metrics_values)
        self.tau_update(self._value_twin, self._target_value_twin)

    def train_policy(self, states):
        vars = self._policy.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(vars)
            values = self._policy_value(states)
            tf.summary.histogram('Policy value', values)
            tf.summary.scalar('Policy value mean', np.mean(values))
            tf.summary.scalar('Policy value std', np.std(values))
            loss = -tf.reduce_mean(values)
        tf.summary.scalar('policy.loss', loss, description='Policy loss, that is negative mean estimated return')
        grads = tape.gradient(loss, vars)
        self._policy_value_optimizer.apply_gradients(zip(grads, vars))
        self.tau_update(self._policy, self._target_policy)

    def unwrap_transitions(self, transitions):
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        returns = self.compute_target_returns(rewards, next_states, dones)
        return states, actions, returns

    def compute_target_returns(self, rewards, next_states, dones):
        next_states = np.asarray(next_states, dtype=np.float32)
        next_actions = self._target_policy.predict_on_batch(next_states)
        next_returns_0 = self._target_value.predict_on_batch({'state': next_states, 'action': next_actions})
        next_returns_1 = self._target_value_twin.predict_on_batch({'state': next_states, 'action': next_actions})
        next_returns = np.minimum(next_returns_0, next_returns_1)
        assert next_returns.shape == (next_states.shape[0], 1)
        next_returns = next_returns[:, 0]  # Flatten
        rewards = np.asarray(rewards, dtype=np.float32)
        returns = rewards + np.logical_not(dones) * self.gamma * next_returns
        return returns

    def predict_actions(self, states):
        return self._policy.predict_on_batch(np.asarray(states, dtype=np.float32))

    def load(self, filepath):
        self._checkpoint.restore(filepath)
        self.loaded = True
        logging.info(f'Checkpoint restored: {filepath}')

    def save(self, filepath):
        path = self._checkpoint.write(filepath)
        logging.info(f'Checkpoint saved: {path}')
        return path


def evaluate(env, network, episodes=100, render_each=None, final_evaluation=False):
    # TODO: Parallelize.
    returns = []
    for _ in range(episodes):
        state, done = env.reset(final_evaluation), False
        rewards = []
        while not done:
            if render_each and env.episode % render_each == 0:
                env.render()
            action = network.predict_actions([state])[0]
            assert action.shape == tuple(env.action_shape)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
        returns.append(np.sum(rewards))
        if render_each and env.episode % render_each == 0:
            logging.debug(f'Rendered episode stats: steps: {len(rewards)}, return: {np.sum(rewards)}')
    tf.summary.histogram('Returns actual', returns, description='Non-discounted returns')
    if len(returns) > 0:
        tf.summary.scalar('Returns actual mean', np.mean(returns), description='Non-discounted returns')
    return returns


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, mu, theta, sigma):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self, mask=None):
        if mask is None:
            mask = np.zeros_like(self.mu)
        self.state = self.state * ~mask + self.mu * mask

    def sample(self):
        assert self.state.shape == self.mu.shape
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


if __name__ == '__main__':
    # Inspired by ddpg.py

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--env", default="BipedalWalker-v2", type=str, help="Environment.")
    # Maximum episode length: 1600
    parser.add_argument("--evaluate_each", default=1600, type=int, help="Evaluate each number of batches.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--noise_sigma", default=0.5, type=float, help="UB noise sigma.")
    parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=128, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--target_tau", default=0.01, type=float, help="Target network update weight.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--workers", default=16, type=int, help="Number of parallel workers.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--replay_buffer_size", default=16 * 1600 * 4, type=int)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--bootstrap", default=0, type=int, help="Choose action uniformly for this many steps.")
    parser.add_argument("--render_final", action="store_true")
    parser.add_argument("--noise_type", choices=["normal", "Ornstein–Uhlenbeck"], default="normal")
    parser.add_argument("--policy_update_period", default=2, type=int)
    parser.add_argument("--input_network", default="walker_network")
    parser.add_argument("--log_dir", default="logs/walker")
    parser.add_argument("--steps", default=1000000, type=int)
    parser.add_argument("--retrain", action="store_true")
    # TODO: Consider early resetting stall episodes.
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    np.seterr(all='raise')

    assert args.batch_size <= args.replay_buffer_size

    logging.debug(tf.config.experimental.list_physical_devices())

    logging.info('Initializing...')

    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    assert tf.executing_eagerly()

    run_name = datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.trace:
        tf.summary.trace_on()
    log_dir = os.path.join(args.log_dir, args.env, run_name)
    writer_train = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    with writer_train.as_default():
        tf.summary.text('args', str(args), step=0)
    writer_evaluate = tf.summary.create_file_writer(os.path.join(log_dir, 'evaluate'))


    def new_environment():
        return gym_evaluator.GymEnvironment(args.env)


    env = new_environment()

    try:
        network = Network(env, args)
        network.load(args.input_network)
        logging.info('Network loaded.')
    except tf.errors.NotFoundError:
        logging.info('Network loading failed gracefully.')

    network_output_path = os.path.join('walker', args.env, run_name)

    if not network.loaded or args.retrain:
        try:
            logging.info('Training...')
            noise = None
            if args.noise_type == 'Ornstein–Uhlenbeck':
                noise = OrnsteinUhlenbeckNoise(np.zeros(tuple([args.workers] + env.action_shape), dtype=np.float32),
                                               args.noise_theta, args.noise_sigma)
            if args.workers > 1:
                states = env.parallel_init(args.workers)
            else:
                states = [env.reset()]
            worker_episode_start_step = np.zeros(args.workers, dtype=np.uint)
            worker_cur_return = np.zeros(args.workers, dtype=np.float)
            worker_prev_return = np.full(args.workers, np.nan, dtype=np.float)
            worker_prev_length = np.full(args.workers, np.nan, dtype=np.float)
            replay_buffer = deque(maxlen=args.replay_buffer_size)

            try:
                import tqdm

                t = tqdm.tqdm(total=args.steps, unit='step')
            except ModuleNotFoundError:
                pass

            stats = dict()
            best_evaluate_return = None
            best_train_return = None
            for step in range(args.steps):
                tf.summary.experimental.set_step(step)
                if args.evaluate_each and step % args.evaluate_each == 0:
                    with writer_evaluate.as_default():
                        returns = evaluate(new_environment(), network, episodes=args.evaluate_for,
                                           render_each=args.render_each)
                        if len(returns) > 0:
                            return_mean = np.mean(returns)
                            stats['evaluate.return'] = return_mean
                            if best_evaluate_return is None or return_mean > best_evaluate_return:
                                best_evaluate_return = return_mean
                                network.save(os.path.join(network_output_path, f'evaluate_{int(return_mean)}'))
                actions_predicted = None
                noise_sample = None
                if step >= args.bootstrap:
                    actions_predicted = network.predict_actions(states)
                    if noise is not None:
                        noise_sample = noise.sample()
                    else:
                        noise_sample = np.random.normal(scale=args.noise_sigma, size=actions_predicted.shape)
                    assert actions_predicted.shape == noise_sample.shape
                    actions_noised = np.clip(actions_predicted + noise_sample, env.action_ranges[0],
                                             env.action_ranges[1])
                else:
                    actions_noised = np.random.uniform(env.action_ranges[0], env.action_ranges[1],
                                                       tuple([args.workers] + env.action_shape))
                assert actions_noised.shape == tuple([args.workers] + env.action_shape)
                assert np.all(actions_noised >= env.action_ranges[0])
                assert np.all(actions_noised <= env.action_ranges[1])
                if args.workers > 1:
                    steps = env.parallel_step(actions_noised)
                else:
                    assert actions_noised.shape[0] == 1
                    steps = [env.step(actions_noised[0])]
                next_states, rewards, dones, _ = zip(*steps)
                if args.workers <= 1 and dones[0]:
                    assert len(dones) == 1
                    env.reset()
                rewards_shaped = []
                for i in range(args.workers):
                    worker_cur_return[i] += rewards[i]
                    if dones[i]:
                        worker_prev_return[i] = worker_cur_return[i]
                        worker_prev_length[i] = step - worker_episode_start_step[i]
                        worker_cur_return[i] = 0
                        worker_episode_start_step[i] = step
                    # rewards_shaped.append(rewards[i] + (max(0, next_states[i][2]) * 4) ** 2)
                    rewards_shaped.append(rewards[i])
                replay_buffer.extend(
                    map(lambda x: Transition(*x), zip(states, actions_noised, rewards_shaped, next_states, dones)))
                with writer_train.as_default():
                    tf.summary.histogram('Velocity X', [state[2] for state in states],
                                         description='Horizontal velocity in a step')
                    if noise_sample is not None:
                        tf.summary.scalar('Noise sample', noise_sample.flatten()[0])
                        tf.summary.histogram(f'Actions noise', noise_sample, description='Actions noise in a step')
                    if actions_predicted is not None:
                        for i in range(actions_predicted.shape[1]):
                            tf.summary.histogram(f'Actions predicted {i}', actions_predicted[:, i],
                                                 description='Actions predicted in a step')
                    for i in range(actions_noised.shape[1]):
                        tf.summary.histogram(f'Actions performed {i}', actions_noised[:, i],
                                             description='Actions performed (noised) in a step')
                    tf.summary.histogram('Rewards', rewards, description='Rewards in a step')
                    stats['reward'] = np.mean(rewards)
                    tf.summary.histogram('Rewards shaped', rewards_shaped, description='Shaped rewards in a step')
                    tf.summary.histogram('Rewards shaping difference', np.asarray(rewards_shaped) - np.asarray(rewards),
                                         description='Reward shaping offset in a step')
                    if not np.isnan(worker_prev_return).any():
                        tf.summary.histogram('Returns actual', worker_prev_return, description='Non-discounted returns')
                        return_mean = worker_prev_return.mean()
                        tf.summary.scalar('Returns actual mean', return_mean, description='Non-discounted returns')
                        stats['train.return'] = worker_prev_return.mean()
                        if step >= args.bootstrap and (best_train_return is None or return_mean > best_train_return):
                            best_train_return = return_mean
                            network.save(os.path.join(network_output_path, f'train_{int(return_mean)}'))
                    if not np.isnan(worker_prev_length).any():
                        tf.summary.histogram('Episode length', worker_prev_length)
                        tf.summary.scalar('Episode length mean', worker_prev_length.mean())
                        stats['train.episode_length'] = worker_prev_length.mean()
                if len(replay_buffer) >= args.batch_size:
                    indexes = np.random.choice(len(replay_buffer), size=args.batch_size)
                    states, actions, returns = network.unwrap_transitions(replay_buffer[i] for i in indexes)
                    with writer_train.as_default():
                        tf.summary.histogram('Actions', actions, description='Actions performed in a batch')
                        tf.summary.histogram('Returns estimated', returns,
                                             description='Estimated returns of state-action pairs in a batch')
                        tf.summary.scalar('Returns estimated mean', returns.mean(),
                                          description='Estimated returns of state-action pairs in a batch')
                        tf.summary.scalar('Returns estimated std', returns.std(),
                                          description='Estimated returns of state-action pairs in a batch')
                        network.train_value(states, actions, returns)
                        if step % args.policy_update_period == 0:
                            network.train_policy(states)
                states = next_states
                try:
                    t.set_postfix(stats)
                    t.update()
                except UnboundLocalError:
                    pass
        finally:
            if args.trace:
                with writer_train.as_default():
                    tf.summary.trace_export('trace')
            network.save(os.path.join(network_output_path, 'final'))
            if args.render_final:
                evaluate(new_environment(), network, episodes=1, render_each=1)
            try:
                t.close()
            except UnboundLocalError:
                pass

    logging.info('Final evaluation...')
    evaluate(new_environment(), network, render_each=args.render_each, final_evaluation=True)
