"""library file to provide constants to get rid of magic numbers"""
import pygame

from ai_wars.enums import EnumAction

# Game constants
BULLET_SPEED = 400.0
SHOOT_COOLDOWN = 200  # specifies the cooldown for shooting in ms
SHIP_SPEED = 200.0
ROTATION_SPEED = 300.0
START_SCORE = 1000  # start score of every player
HITSCAN_ENABLED = True
RELATIVE_COORDINATES_MODE = True

# Server constants
SERVER_TICK_RATE = 30
SERVER_TIMEOUT = 1.0
POINTS_LOST_AFTER_GETTING_HIT = 100
POINTS_GAINED_AFTER_HITTING = 200

# Client constants
HEIGHT = 600
WIDTH = 800
CLIENT_BUFFER_SIZE = 10 * 1024
# http://phrogz.net/tmp/24colors.html
COLOR_ARRAY = [[255, 0, 0], [237, 185, 185], [143, 35, 35], [255, 255, 0], [185, 215, 237], [35, 98, 143], [115, 115, 115],
               [0, 234, 255], [231, 233, 185], [143, 106, 35], [204, 204, 204], [170, 0, 255], [220, 185, 237],
               [255, 127, 0], [185, 237, 224], [79, 143, 35], [191, 255, 0], [0, 149, 255], [255, 0, 170], [255, 212, 0],
               [106, 255, 0], [0, 64, 255]]

# pygame userevents use codes from 24 to 35, so the first user event will be 24
DECREASE_SCORE_EVENT = pygame.USEREVENT + 0  # event code 24

# DQN gamestate constants
MODEL_PATH = "models/"
MAX_NUM_PROJECTILES = 128
NUM_PLAYERS = 2
MOVEMENT_SET = EnumAction

# DQN agent constants
MEMORY_SIZE = 10000
BATCH_SIZE = 64

GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
DECAY_FACTOR = 0.99995

LSTM_SEQUENCE_SIZE = 32
