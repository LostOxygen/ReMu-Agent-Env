import abc

import pygame
from pygame.math import Vector2

class Checkpoint:
    def __init__(self, checkpoint_rect: pygame.rect, color: (int,int,int)):
        self.rect = checkpoint_rect
        self.middle_point = Vector2(checkpoint_rect.x + checkpoint_rect.width / 2,
                                    checkpoint_rect.y + checkpoint_rect.height / 2)
        self.color = color

class Map(abc.ABC):
    '''
    Abstract class for all maps.
    '''

    def __init__(self, screen: pygame.surface):
        # Define and add to these properties in the subclasses of map (Actual maps)
        self.bound_rects: list[pygame.rect] = []
        self.checkpoints: list[Checkpoint] = []
        self.goal: Checkpoint = None
        self.spawn_point = None
        self.spawn_direction = None
        self.screen = screen

    def draw(self) -> None:
        # draw bounds
        for rect in self.bound_rects:
            pygame.draw.rect(self.screen, 'gray', rect)

        # draw checkpoints
        for checkpoint in self.checkpoints:
            pygame.draw.rect(self.screen, checkpoint.color, checkpoint.rect)

        # draw goal
        pygame.draw.rect(self.screen, self.goal.color, self.goal.rect)

    def is_point_in_bounds(self, point: Vector2) -> bool:
        for rect in self.bound_rects:
            if rect.collidepoint(point):
                return False

        return True

    def is_point_on_checkpoints(self, point: Vector2) -> bool:
        for checkpoint in self.checkpoints:
            if checkpoint.rect.collidepoint(point):
                return True

        return False