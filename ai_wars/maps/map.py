import abc

import pygame
from pygame.math import Vector2


class Map(abc.ABC):
    '''
    Abstract class for all maps.
    '''
    def __init__(self, screen: pygame.surface):
        self.bound_rects: list[pygame.rect] = []
        self.checkpoints: list[pygame.rect] = []
        self.screen = screen
        pass

    def draw(self) -> None:
        # draw bounds
        for rect in self.bound_rects:
            pygame.draw.rect(self.screen, 'gray', rect)

        # draw checkpoints
        for checkpoint in self.checkpoints:
            pygame.draw.rect(self.screen, 'orange', checkpoint)

        # draw goal
        pygame.draw.rect(self.screen, 'white', self.goal_rect)

    def is_point_in_bounds(self, point: Vector2) -> bool:
        for rect in self.bound_rects:
            if rect.collidepoint(point):
                return False

        return True

    def is_point_on_checkpoints(self, point: Vector2) -> bool:
        for checkpoint in self.checkpoints:
            if checkpoint.collidepoint(point):
                return True

        return False
