"""Bullet class file"""
import pygame
from pygame.math import Vector2


class Bullet():
	"""Bullet class with functions for moving and drawing"""
	def __init__(self, x: int, y: int, height: int, width: int,
				 sprite: pygame.Surface, velocity: Vector2):
		self.x = x
		self.y = y
		self.height = height
		self.width = width
		self.sprite = sprite
		self.velocity = velocity

	def move(self) -> None:
		self.x = self.x + self.velocity.x
		self.y = self.y + self.velocity.y

	def draw(self, surface: pygame.Surface) -> None:
		# bullet_position = pygame.Rect(self.x, self.y, self.height, self.width)
		surface.blit(self.sprite, (self.x, self.y))
