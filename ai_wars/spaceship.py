"""spacehip class file"""
from typing import Callable
import pygame
from pygame.math import Vector2

from ai_wars.enums import EnumAction
from ai_wars.bullet import Bullet
from ai_wars.utils import load_sprite

UP = Vector2(0, -1)


class Spaceship():
	"""spaceship class with functions for moving, drawing and shooting"""
	def __init__(self, x: int, y: int, height: int, width: int,
				 sprite: pygame.sprite.Sprite, bullet_append_func: Callable[[Bullet], None],
				 surface: pygame.Surface):
		self.x = x
		self.y = y
		self.height = height
		self.width = width
		self.sprite = sprite
		self.bullet_append = bullet_append_func
		self.direction = Vector2(UP)
		self.surface = surface
		self.velocity = 2

	def draw(self, surface: pygame.Surface) -> None:
		#player_rec = pygame.Rect(self.x, self.y, self.height, self.width)
		surface.blit(self.sprite, (self.x, self.y))

	def move(self, action: EnumAction) -> None:
		if action == EnumAction.FORWARD and self.y - self.velocity > 0:
			self.y -= self.velocity

		if action == EnumAction.BACKWARDS and self.y + self.velocity < \
				(self.surface.get_height() - self.height):
			self.y += self.velocity

		if action == EnumAction.RIGHT and self.x + self.velocity < \
				(self.surface.get_width() - self.width):
			self.x += self.velocity

		if action == EnumAction.LEFT and self.x - self.velocity > 0:
			self.x -= self.velocity

	def shoot(self) -> None:
		bullet_velocity = self.direction * 3
		bullet = Bullet(self.x, self.y, self.height, self.width,
						load_sprite("ai_wars/img/bullet.png"), bullet_velocity)
		self.bullet_append(bullet)
		