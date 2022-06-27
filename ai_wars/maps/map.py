import abc

import pygame
from pygame.math import Vector2

class Goal:
    def __init__(self, goal_rect: pygame.rect):
        self.goal_rect = goal_rect
        self.middle_point = Vector2(goal_rect.x + goal_rect.width / 2,
                                    goal_rect.y + goal_rect.height / 2)

class Checkpoint:
    def __init__(self, checkpoint_rect: pygame.rect):
        self.checkpoint_rect = checkpoint_rect
        self.middle_point = Vector2(checkpoint_rect.x + checkpoint_rect.width / 2,
                                    checkpoint_rect.y + checkpoint_rect.height / 2)


class Map(abc.ABC):
    '''
    Abstract class for all maps.
    '''

    def __init__(self, screen: pygame.surface):
        # Define and add to these properties in the subclasses of map (Actual maps)
        self.bound_rects: list[pygame.rect] = []
        self.checkpoints: list[Checkpoint] = []
        self.goal: Goal = None
        self.spawn_point = None
        self.spawn_direction = None
        self.screen = screen
        pass

    def draw(self) -> None:
        # draw bounds
        for rect in self.bound_rects:
            pygame.draw.rect(self.screen, 'gray', rect)

        # draw checkpoints
        for checkpoint in self.checkpoints:
            pygame.draw.rect(self.screen, 'orange', checkpoint.checkpoint_rect)

        # draw goal
        pygame.draw.rect(self.screen, 'white', self.goal.goal_rect)

    def is_point_in_bounds(self, point: Vector2) -> bool:
        for rect in self.bound_rects:
            if rect.collidepoint(point):
                return False

        return True

    def is_point_on_checkpoints(self, point: Vector2) -> bool:
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_rect.collidepoint(point):
                return True

        return False
