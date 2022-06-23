import abc

import pygame
from pygame.math import Vector2


class Map(abc.ABC):
    '''
    Abstract class for all maps.
    '''
    def __init__(self, screen: pygame.surface):
        self.boundRects: list[pygame.rect] = []
        self.screen = screen
        pass

    def draw(self) -> None:
        # draw bounds
        for rect in self.boundRects:
            pygame.draw.rect(self.screen, 'gray', rect)

        # draw goal
        pygame.draw.rect(self.screen, 'white', self.goalRect)

    def is_point_in_bounds(self, point: Vector2) -> bool:
        for rect in self.boundRects:
            if rect.collidepoint(point):
                return False

        return True
