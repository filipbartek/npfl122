#!/usr/bin/env python3
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses

import az_quiz

_BOARD_FIRST_PLAYER = 2
_BOARD_SECOND_PLAYER = 3
RECTANGLE_BOARD_WIDTH = 13
RECTANGLE_BOARD_HEIGHT = 7
N_PLANES_PER_PLAYER = 6
ACTION_SIZE = 28
L2_REGULARIZATION = 1e-4

class AlphaZeroConfig(object):

  def __init__(self):
    ### Self-Play
    self.num_actors = 5000

    self.num_sampling_moves = 30
    self.num_simulations = 800

    # Root prior exploration noise.
    self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    ### Training
    self.training_steps = int(700e3)
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = 4096

    self.weight_decay = 1e-4
    self.momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    self.learning_rate_schedule = {
        0: 2e-1,
        100e3: 2e-2,
        300e3: 2e-3,
        500e3: 2e-4
    }

class Node(object):
  def __init__(self, prior):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}

  def expanded(self):
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

class Game(object):
    def __init__(self, history, az_quiz = None):
        self.history = history or []
        self.child_visits = []
        self.num_actions = 28
        self.az_quiz = az_quiz.AZQuiz(False) if az_quiz == None else az_quiz

    def terminal(self):
        # Game specific termination rules.
        return self.az_quiz.winner != None
        # TODO: what happens if the board is full and yet nobody wins?

    def terminal_value(self, to_play):
        # Game specific value.
        if self.az_quiz.winner == 0: # assuming we are the first player
            return 1
        elif self.az_quiz.winner == 1:
            return -1

    def legal_actions(self):
        # Game specific calculation of legal actions.
        result = []
        for action in range(self.num_actions):
            if self.az_quiz.valid(action):
                result.append(action)
        return result

    def clone(self):
        return Game(list(self.history))

    def apply(self, action):
        self.history.append(action)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.itervalues())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index):
        nPlanesPerPlayer = N_PLANES_PER_PLAYER
        width = RECTANGLE_BOARD_WIDTH
        height = RECTANGLE_BOARD_HEIGHT
        image = np.full((nPlanesPerPlayer+nPlanesPerPlayer, width, height), False, np.bool)
        state = self.reconstruct_state_from_history_and_state_index(state_index)
        for i in range(self.num_actions):
            j, k = self.get_rectangular_index_from_action_index(i)
            if state[i] == _BOARD_FIRST_PLAYER:
                image[0:nPlanesPerPlayer, j, k] = True
            elif state[i] == _BOARD_SECOND_PLAYER:
                image[nPlanesPerPlayer:2*nPlanesPerPlayer, j, k] = True
        return image # Game specific feature planes

    def get_rectangular_index_from_action_index(self, action_index):
        mapping = [(6,0),
                    (5,1), (7,1),
                    (4,2), (6,2), (8,2),
                    (3,3), (5,3), (7,3), (9,3),
                    (2,4), (4,4), (6,4), (8,4), (10,4),
                    (1,5), (3,5), (5,5), (7,5), (9,5), (11,5),
                    (0,6), (2,6), (4,6), (6,6), (8,6), (10,6), (12,6)]
        return mapping[action_index]

    def player_to_board_player(self, player):
        return _BOARD_FIRST_PLAYER if player == 0 else _BOARD_SECOND_PLAYER

    def reconstruct_state_from_history_and_state_index(self, state_index):
        player = 0
        state = np.zeros((self.num_actions), np.uint8)
        for i in range(state_index+1):
            state[self.history[i]] = self.player_to_board_player(player)
            player = (player + 1) % 2
        return state

    def make_target(self, state_index):
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    def to_play(self):
        return len(self.history) % 2

class ReplayBuffer(object):

  def __init__(self, config: AlphaZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self):
    # Sample uniformly across positions.
    move_sum = float(sum(len(g.history) for g in self.buffer))
    games = np.random.choice(
        self.buffer,
        size=self.batch_size,
        p=[len(g.history) / move_sum for g in self.buffer])
    game_pos = [(g, np.random.randint(len(g.history))) for g in games]
    return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]

# Disclaimer: based on https://github.com/suragnair/alpha-zero-general/blob/master/othello/tensorflow/OthelloNNet.py
class Network(object):
    def __init__(self):
        nPlanesPerPlayer = N_PLANES_PER_PLAYER
        width = RECTANGLE_BOARD_WIDTH
        height = RECTANGLE_BOARD_HEIGHT

        input = tf.keras.Input(shape=(nPlanesPerPlayer*2, width, height), dtype=tf.bool)
        x_image = tf.reshape([-1, width, height, 1])(input) # batch_size  x board_x x board_y x 1
        regularizer = tf.keras.regularizers.l2(l=L2_REGULARIZATION)
        x_image = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1),name='conv',padding='same',use_bias=False, kernel_regularizer=regularizer)(x_image)
        x_image = layers.BatchNormalization(axis=1, name='conv_bn')(x_image)
        x_image = tf.nn.relu(x_image)

        num_channels = 256

        residual_tower = self.residual_block(inputLayer=x_image, kernel_size=3, filters=num_channels, stage=1, block='a')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=2, block='b')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=3, block='c')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=4, block='d')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=5, block='e')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=6, block='g')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=7, block='h')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=8, block='i')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=9, block='j')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=10, block='k')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=11, block='m')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=12, block='n')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=13, block='o')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=14, block='p')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=15, block='q')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=16, block='r')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=17, block='s')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=18, block='t')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=19, block='u')

        policy = layers.Conv2D(2,kernel_size=(1, 1), strides=(1, 1),name='policy',padding='same',use_bias=False, kernel_regularizer=regularizer)(residual_tower)
        policy = layers.BatchNormalization(axis=3, name='bn_policy')(policy)
        policy = tf.nn.relu(policy)
        policy = layers.Flatten(name='p_flatten')(policy)
        self.policy = layers.Dense(ACTION_SIZE, kernel_regularizer=regularizer)(policy)
        self.prob = tf.nn.softmax(self.policy)

        value = layers.Conv2D(1,kernel_size=(1, 1), strides=(1, 1),name='value',padding='same',use_bias=False, kernel_regularizer=regularizer)(residual_tower)
        value = layers.BatchNormalization(axis=3, name='bn_v')(value)
        value = tf.nn.relu(value)
        value = layers.Flatten(name='v_flatten')(value)
        value = layers.Dense(units=256, kernel_regularizer=regularizer)(value)
        value = tf.nn.relu(value)
        value = layers.Dense(1, kernel_regularizer=regularizer)(value)
        self.v = tf.nn.tanh(value) 
    
    def residual_block(self,inputLayer, filters,kernel_size,stage,block):
        conv_name = 'res' + str(stage) + block + '_branch'
        bn_name = 'bn' + str(stage) + block + '_branch'

        shortcut = inputLayer

        regularizer = tf.keras.regularizers.l2(l=L2_REGULARIZATION)
        residual_layer = layers.Conv2D(filters,kernel_size=(kernel_size, kernel_size), strides=(1, 1),name=conv_name+'2a',padding='same',use_bias=False, kernel_regularizer=regularizer)(inputLayer)
        residual_layer = layers.BatchNormalization(axis=3, name=bn_name+'2a')(residual_layer)
        residual_layer = tf.nn.relu(residual_layer)
        residual_layer = layers.Conv2D(filters,kernel_size=(kernel_size, kernel_size), strides=(1, 1),name=conv_name+'2b',padding='same',use_bias=False, kernel_regularizer=regularizer)(residual_layer)
        residual_layer = layers.BatchNormalization(axis=3, name=bn_name+'2b')(residual_layer)
        add_shortcut = tf.add(residual_layer, shortcut)
        residual_result = tf.nn.relu(add_shortcut)
        
        return residual_result

    def inference(self, image):
        value = self.v.predict_on_batch(image)
        policy = self.policy.predict_on_batch(image)
        return value, policy

    #def get_weights(self):
    #    # Returns the weights of this network.
    #    return []

class SharedStorage(object):

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      return make_uniform_network()  # policy -> uniform, value -> 0.5

  def save_network(self, step: int, network: Network):
    self._networks[step] = network

def make_uniform_network():
    return Network()

# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphazero(config: AlphaZeroConfig):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    nIterations = 100
    for j in range(nIterations):
        for i in range(config.num_actors):
            launch_job(run_selfplay, config, storage, replay_buffer)

        train_network(config, storage, replay_buffer)

    return storage.latest_network()

def launch_job(f, *args):
    f(*args)

##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)

# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: Network):
  game = Game(None)
  while not game.terminal() and len(game.history) < config.max_moves:
    action, root = run_mcts(config, game, network)
    game.apply(action)
    game.store_search_statistics(root)
  return game

# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config, game, network):
  root = Node(0)
  evaluate(root, game, network)
  add_exploration_noise(config, root)

  for _ in range(config.num_simulations):
    node = root
    scratch_game = game.clone()
    search_path = [node]

    while node.expanded():
      action, node = select_child(config, node)
      scratch_game.apply(action)
      search_path.append(node)

    value = evaluate(node, scratch_game, network)
    backpropagate(search_path, value, scratch_game.to_play())
  return select_action(config, game, root), root

def select_action(config, game, root):
  visit_counts = [(child.visit_count, action)
                  for action, child in root.children.iteritems()]
  if len(game.history) < config.num_sampling_moves:
    _, action = softmax_sample(visit_counts)
  else:
    _, action = max(visit_counts)
  return action

def softmax_sample(visit_counts):
    counts_exp = np.exp(visit_counts)
    probs = counts_exp / np.sum(counts_exp, axis=0)
    return np.random.choice(len(ACTION_SIZE), p=probs)

# Select the child with the highest UCB score.
def select_child(config, node):
  _, action, child = max((ucb_score(config, node, child), action, child)
                         for action, child in node.children.iteritems())
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config, parent, child):
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  value_score = child.value()
  return prior_score + value_score

# We use the neural network to obtain a value and policy prediction.
def evaluate(node, game, network):
  value, policy_logits = network.inference(game.make_image(-1))

  # Expand the node.
  node.to_play = game.to_play()
  policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
  policy_sum = sum(policy.itervalues())
  for action, p in policy.iteritems():
    node.children[action] = Node(p / policy_sum)
  return value

# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path, value, to_play):
  for node in search_path:
    node.value_sum += value if node.to_play == to_play else (1 - value)
    node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config, node):
  actions = node.children.keys()
  noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########
def train_network(config, storage, replay_buffer):
  network = Network()
  optimizer = tf.compat.v1.train.MomentumOptimizer(config.learning_rate_schedule, config.momentum)
  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch()
    update_weights(optimizer, network, batch, config.weight_decay)
  storage.save_network(config.training_steps, network)

def update_weights(optimizer, network, batch, weight_decay):
  loss = 0
  for image, (target_value, target_policy) in batch:
    value, policy_logits = network.inference(image)
    loss += (
        tf.losses.mean_squared_error(value, target_value) +
        tf.nn.softmax_cross_entropy_with_logits(
            logits=policy_logits, labels=target_policy))

  #for weights in network.get_weights():
  #  loss += weight_decay * tf.nn.l2_loss(weights)

  optimizer.minimize(loss)

class Player:
    def __init__(self):
        self.config = AlphaZeroConfig()
        self.net = alphazero(self.config)

    def play(self, az_quiz):
        game = Game(None, az_quiz)
        action = run_mcts(self.config, game, self.net) # TODO: greedily choose the action
        return action

if __name__ == "__main__":
    import az_quiz_evaluator_recodex
    az_quiz_evaluator_recodex.evaluate(Player())
