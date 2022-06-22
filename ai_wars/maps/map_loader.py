import pygame

from ..maps.straight import Straight


def load_map(screen: pygame.surface, name):
    if name == "straight":
        return Straight(screen)
