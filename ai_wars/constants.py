"""library file to provide constants to get rid of magic numbers"""
import pygame

# Game constants
BULLET_SPEED = 1000000.0
SHOOT_COOLDOWN = 200  # specifies the cooldown for shooting in ms
SHIP_SPEED = 200.0
ROTATION_SPEED = 300.0
START_SCORE = 1000  # start score of every player

# Server constants
SERVER_TICK_RATE = 30
SERVER_TIMEOUT = 1.0
POINTS_LOST_AFTER_GETTING_HIT = 100
POINTS_GAINED_AFTER_HITTING = 200

# Client constants
CLIENT_BUFFER_SIZE = 10 * 1024

# pygame userevents use codes from 24 to 35, so the first user event will be 24
DECREASE_SCORE_EVENT = pygame.USEREVENT + 0  # event code 24

# DQN gamestate constants
MODEL_PATH = "models/"
MAX_NUM_PROJECTILES = 128
NUM_PLAYERS = 2

# DQN agent constants
MEMORY_SIZE = 10000
BATCH_SIZE = 64

GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
DECAY_FACTOR = 0.99995

LSTM_SEQUENCE_SIZE = 32
