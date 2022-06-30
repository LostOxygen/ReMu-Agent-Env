import pygame

from ..maps.straight import Straight
from ..maps.corner import Corner


def load_map(screen: pygame.surface, name):
	if name == "straight":
		return Straight(screen)

	if name == "corner":
		return Corner(screen)
