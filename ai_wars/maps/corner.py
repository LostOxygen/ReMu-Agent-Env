import pygame

from .map import Map,Checkpoint
from ..utils import override
from pygame import Rect
from pygame import Vector2

class Corner(Map):
    '''
    A map that is only a straight line from left to right
    '''

    def __init__(self, screen: pygame.surface):
        super().__init__(screen)

        # Create map boundaries
        self.bound_rects.append(Rect(0, 0, 400, 350))
        self.bound_rects.append(Rect(550, 0, 250, 500))
        self.bound_rects.append(Rect(0, 500, 800, 100))

        # Create checkpoints
        self.checkpoints.append(Checkpoint(Rect(390, 350, 10, 150), 'orange'))

        # Create Goal
        self.goal = Checkpoint(Rect(0, 350, 10, 150), 'white')
        # Goal is also a checkpoint
        self.checkpoints.append(self.goal)

        # Spawn
        self.spawn_point = Vector2(475, 0)
        self.spawn_direction = Vector2(0, 1)

        # Needed for calculation
        self.max_dist_between_spawn_and_goal = self.spawn_point.distance_squared_to(self.goal.middle_point)
