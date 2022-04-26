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
		self.acceleration = 0.9
		self.draw_position = Vector2() # position where the spaceship gets drawn

	def move(self, action: EnumAction) -> None:
		"""public method to move the ship in the direction of the action"""
		if action == EnumAction.FORWARD and self.y - self.velocity > 0:
			self.y += self.direction.y * self.acceleration
			self.x += self.direction.x * self.acceleration

		if action == EnumAction.BACKWARDS and self.y + self.velocity < \
				(self.surface.get_height() - self.height):
			self.y -= self.direction.y * self.acceleration
			self.x -= self.direction.x * self.acceleration

		if action == EnumAction.RIGHT and self.x + self.velocity < \
				(self.surface.get_width() - self.width):
			self.rotate(clockwise=True)

		if action == EnumAction.LEFT and self.x - self.velocity > 0:
			self.rotate(clockwise=False)

	def shoot(self) -> None:
		"""public method to create a bullet and append it to the bullet list"""
		bullet_velocity = self.direction * 3
		bullet = Bullet(self.name, self.x+np.floor(self.width/2)-5, self.y, self.height, self.width, \
						load_sprite("ai_wars/img/bullet.png"), bullet_velocity)
		self.bullet_append(bullet)

	def got_hit(self) -> None:
		"""public method to decrease the score, everytime the ship got hit"""
		self.score -= 10

	def rotate(self, clockwise=True) -> None:
		"""public method to rotate the ship in clockwise direction"""
		# rotate the direction vector
		sign = 1 if clockwise else -1
		angle = 3 * sign
		self.direction.rotate_ip(angle)

	def draw(self, surface: pygame.Surface) -> None:
		# player_rec = pygame.Rect(self.x, self.y, self.height, self.width)
		# surface.blit(self.sprite, (self.x, self.y))
		angle = self.direction.angle_to(UP)
		rotated_surface = pygame.transform.rotozoom(self.sprite, angle, 1.0)
		rotated_surface_size = Vector2(rotated_surface.get_size())
		blit_position = Vector2(self.x, self.y) - rotated_surface_size * 0.5
		surface.blit(rotated_surface, blit_position)
