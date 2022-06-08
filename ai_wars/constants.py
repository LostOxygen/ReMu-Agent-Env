"""library file to provide constants to get rid of magic numbers"""
import pygame

from ai_wars.enums import EnumAction, RotationOnlyActions  # pylint: disable=unused-import

# Game constants
BULLET_SPEED = 400.0
SHOOT_COOLDOWN = 1000  # specifies the cooldown for shooting in ms
SHIP_SPEED = 150.0
ROTATION_SPEED = 100.0
START_SCORE = 10000  # start score of every player
HITSCAN_ENABLED = False
POINTS_LOST_PER_SECOND = 0

# Server constants
SERVER_TICK_RATE = 30
SERVER_TIMEOUT = 1.0
POINTS_LOST_AFTER_GETTING_HIT = 100
POINTS_GAINED_AFTER_HITTING = 200

# Client constants
HEIGHT = 600
WIDTH = 800
CLIENT_BUFFER_SIZE = 10 * 1024

# pygame userevents use codes from 24 to 35, so the first user event will be 24
DECREASE_SCORE_EVENT = pygame.USEREVENT + 0  # event code 24

# DQN gamestate constants
MODEL_PATH = "models/"

MAX_NUM_PROJECTILES = 0 # the number of projectiles in the gamestate tensor
NUM_PLAYERS = 8
MOVEMENT_SET = RotationOnlyActions  # pylint: disable=invalid-name

# DQN agent constants
MEMORY_SIZE = int(1e5)
BATCH_SIZE = 64

# DQN hyperparameters
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
DECAY_FACTOR = 0.99995
UPDATE_EVERY = 100
USE_REPLAY_AFTER = 10000
LEARNING_RATE = 0.001
TAU = 1e-3

LSTM_SEQUENCE_SIZE = 32
