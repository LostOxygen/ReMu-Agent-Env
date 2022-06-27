import pygame

from .map import Map, Checkpoint
from ..utils import override
from pygame import Rect
from pygame import Vector2

class Straight(Map):
    '''
    A map that is only a straight line from left to right
    '''

    def __init__(self, screen: pygame.surface):
        super().__init__(screen)

        # Create map boundaries
        self.bound_rects.append(Rect(0, 0, 800, 200))
        self.bound_rects.append(Rect(0, 400, 800, 200))

        # Goal properties
        self.goal = Checkpoint(Rect(790, 200, 10, 200), 'white')

        # Spawn properties
        self.spawn_point = Vector2(0, 300)
        self.spawn_direction = Vector2(1, 0)

        # Needed for calculation
        self.max_dist_between_spawn_and_goal = self.spawn_point.distance_squared_to(self.goal.middle_point)
