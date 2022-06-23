import pygame

from .map import Map
from ..utils import override
from pygame import Rect
from pygame import Vector2

class Straight(Map):
    '''
    A map that is only a straight line from left to right
    '''

    def __init__(self, screen: pygame.surface):
        super().__init__(screen)
        self.boundRects.append(Rect(0, 0, 800, 200))
        self.boundRects.append(Rect(0, 400, 800, 200))
        self.goalRect = pygame.rect.Rect(780, 200, 10, 200)
        self.goalPoint = Vector2(785, 300)
        pass
