"""spacehip class file"""
from typing import Callable
import pygame
from pygame.math import Vector2
import numpy as np

from ai_wars.enums import EnumAction
from ai_wars.bullet import Bullet
from ai_wars.utils import load_sprite

UP = Vector2(0, -1)


class Spaceship():
	"""spaceship class with functions for moving, drawing and shooting"""

	def __init__(self, x: int, y: int, height: int, width: int, \
				 sprite: pygame.sprite.Sprite, \
				 bullet_append_func: Callable[[Bullet], None], \
				 surface: pygame.Surface, name: str):
		self.x = x
		self.y = y
		self.height = height
		self.width = width
		self.sprite = sprite
		self.bullet_append = bullet_append_func
		self.direction = Vector2(UP)
		self.surface = surface
		self.velocity = 2
		self.score = 1000 # start with a score of 1000 and reduce over time and per hit
		self.name = name # name is equivalent to an player ID

	def move(self, action: EnumAction) -> None:
		"""public method to move the ship in the direction of the action"""
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
		"""public method to create a bullet and append it to the bullet list"""
		bullet_velocity = self.direction * 3
		bullet = Bullet(self.x+np.floor(self.width/2)-5, self.y, self.height, self.width, \
						load_sprite("ai_wars/img/bullet.png"), bullet_velocity)
		self.bullet_append(bullet)

	def got_hit(self) -> None:
		"""public method to decrease the score, everytime the ship got hit"""
		self.score -= 10
