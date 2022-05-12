"""library file to provide constants for important functionalities and to get rid of magic numbers"""
import pygame

# Server constants
SERVER_TICK_RATE = 30
SERVER_TIMEOUT = 1.0
POINTS_LOST_AFTER_GETTING_HIT = 100
POINTS_GAINED_AFTER_HITTING = 200

# Client constants
CLIENT_BUFFER_SIZE = 10 * 1024

# pygame userevents use codes from 24 to 35, so the first user event will be 24
DECREASE_SCORE_EVENT = pygame.USEREVENT + 0  # event code 24
