"""Utils library for miscellaneous functions"""
import random
from typing import Tuple

import torch
import pygame
from pygame.image import load
from pygame.math import Vector2

from .constants import (HEIGHT, WIDTH)


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


def surface_to_tensor(surface: pygame.Surface) -> torch.tensor:
	"""returns the surface as a pytorch tensor of dimension (channels, height, width)"""
	img = torch.tensor(pygame.surfarray.array3d(surface))
	img = img.permute(2, 1, 0) # swap axes to the pytorch order
	return img


def render_to_surface(
    players: list[dict[str, any]],
    projectiles: list[dict[str, any]]
	) -> pygame.Surface:
	"""
	Renders the gamestate to a pygame surface

	Parameters:
		players: The players with their coordinates and directions
		projectiles: The projectiles with their coordinates and directions
		scoreboard: The scoreboard dictionary with the scores of the players
		device: The device the tensor should be stored on (cpu or cuda:0)

	Return:
		pygame.Surface: the gamestate, rendered to a pygame.Surface
	"""
	surface = pygame.Surface((WIDTH, HEIGHT))
	background_sprite = load_sprite("ai_wars/img/Background.png", True)
	spaceship_sprite = load_sprite("ai_wars/img/spaceship.png", True)
	bullet_sprite = load_sprite("ai_wars/img/bullet.png", True)

	# draw the black background
	surface.blit(background_sprite, (0, 0))
	# draw the bullets#
	for projectile in projectiles:
		surface.blit(bullet_sprite, (projectile["position"].x, projectile["position"].y))

	# draw the players
	for player in players:
		angle = player["direction"].angle_to(Vector2(0, -1))
		rotated_surface = pygame.transform.rotozoom(spaceship_sprite, angle, 1.0)
		rotated_surface_size = Vector2(rotated_surface.get_size())
		blit_position = Vector2(player["position"].x, player["position"].y)-rotated_surface_size * 0.5
		surface.blit(rotated_surface, blit_position)

	return surface
