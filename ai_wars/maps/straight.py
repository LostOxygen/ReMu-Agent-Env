import pygame

from .map import Map
from ..utils import override
from pygame import Rect

class Straight(Map):
    '''
    A map that is only a straight line from left to right
    '''

    def __init__(self, screen: pygame.surface):
        super().__init__(screen)
        self.boundRects.append(Rect(0, 0, 800, 200))
        self.boundRects.append(Rect(0, 400, 800, 200))
        pass
