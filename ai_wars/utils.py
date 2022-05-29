"""Utils library for miscellaneous functions"""
import random
from typing import Tuple

import torch
import pygame
from pygame.image import load
from pygame.math import Vector2


def load_sprite(path: str, with_alpha=True) -> pygame.Surface:
	loaded_sprite = load(path)

	if with_alpha:
		return loaded_sprite.convert_alpha()
	else:
		return loaded_sprite.convert()


def wrap_position(position: Tuple[int, int], surface: pygame.Surface) -> Vector2:
	x, y = position
	w, h = surface.get_size()
	return Vector2(x % w, y % h)


def get_random_position(surface: pygame.Surface) -> Vector2:
	return Vector2(
		random.randrange(surface.get_width()),
		random.randrange(surface.get_height()),
	)


def get_random_velocity(min_speed: float, max_speed: float) -> Vector2:
	speed = random.randint(min_speed, max_speed)
	angle = random.randrange(0, 360)
	return Vector2(speed, 0).rotate(angle)


def clip(score: int) -> int:
	return max(score, 0)


def clip_pos(coord: int, min_value: int, max_value: int) -> int:
	return max(min(coord, max_value), min_value)


def override(func):
	'''simple annotation to indicate that a functions overrides an abstract method of its parent'''
	return func


def surface_to_array(surface: pygame.Surface) -> torch.tensor:
	"""returns the surface as a pytorch array of dimension (channels, height, width)"""
	img = torch.tensor(pygame.surfarray.array3d(surface))
	img = img.permute(2, 1, 0) # swap axes to the pytorch order
	return img
