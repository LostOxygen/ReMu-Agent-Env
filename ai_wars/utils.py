"""Utils library for miscellaneous functions"""
from typing import Tuple

import torchvision
import torch
import pygame
from pygame.image import load
from pygame.math import Vector2

from .constants import (HEIGHT, WIDTH)


def load_sprite(path: str, with_alpha: bool=True, is_cnn: bool=False) -> pygame.Surface:
	loaded_sprite = load(path)
	if is_cnn:
		return loaded_sprite
	elif with_alpha:
		return loaded_sprite.convert_alpha()
	else:
		return loaded_sprite.convert()


def wrap_position(position: Tuple[int, int], surface: pygame.Surface) -> Vector2:
	x, y = position
	w, h = surface.get_size()
	return Vector2(x % w, y % h)


def clip(score: int) -> int:
	return max(score, 0)


def clip_pos(coord: int, min_value: int, max_value: int) -> int:
	return max(min(coord, max_value), min_value)


def override(func):
	'''simple annotation to indicate that a functions overrides an abstract method of its parent'''
	return func


def convert_to_greyscale(image: torch.tensor) -> torch.tensor:
	"""
	Converts a pytorch tensor to greyscale

	Parameters:
		image: The image to convert

	Return:
		torch.tensor: The converted image
	"""
	return torchvision.transforms.Grayscale()(image)


def surface_to_tensor(surface: pygame.Surface, device: str) -> torch.tensor:
	"""
	Converts the surface to a pytorch tensor of dimension (channels, height, width)

	Parameters:
		surface: The surface to convert
		device: The device the tensor should be stored on (cpu or cuda:0)

	Returns:
		torch.tensor: The converted surface
	"""
	img_tensor = torch.tensor(pygame.surfarray.array3d(surface), dtype=torch.float, device=device)
	img_tensor = img_tensor.permute(2, 0, 1)  # swap axes to the pytorch order

	return img_tensor


def render_to_surface(players: list[dict[str, any]]) -> pygame.Surface:
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
	spaceship_sprite = load_sprite("ai_wars/img/spaceship.png", True, True)

	# draw the players
	for player in players:
		angle = player["direction"].angle_to(Vector2(0, -1))
		rotated_surface = pygame.transform.rotozoom(spaceship_sprite, angle, 1.0)
		rotated_surface_size = Vector2(rotated_surface.get_size())
		blit_position = Vector2(player["position"].x, player["position"].y)-rotated_surface_size * 0.5
		surface.blit(rotated_surface, blit_position)

		surface = resize_surface(surface, (75, 100))

	return surface


def resize_surface(surface: pygame.Surface, new_size: Tuple[int, int]) -> pygame.Surface:
	"""
	Resizes a surface to a new size

	Parameters:
		surface: The surface to resize
		new_size: The new size of the surface as Tuple of (width, height)

	Return:
		pygame.Surface: The resized surface
	"""
	return pygame.transform.scale(surface, new_size)
