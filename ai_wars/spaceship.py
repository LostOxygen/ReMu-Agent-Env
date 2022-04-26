"""spacehip class file"""
from typing import Callable
import pygame
from pygame.math import Vector2
import numpy as np

from ai_wars.enums import EnumAction
from ai_wars.bullet import Bullet
from ai_wars.utils import load_sprite
from ai_wars.scoreboard import Observer, Subject, Scoreboard

UP = Vector2(0, -1)


class Spaceship(Observer):
	"""spaceship class with functions for moving, drawing and shooting"""

	def __init__(self, x: int, y: int, height: int, width: int, \
				 sprite: pygame.sprite.Sprite, \
				 bullet_append_func: Callable[[Bullet], None], \
              	 screen: pygame.Surface,
				 name: str):
		self.x = x
		self.y = y
		self.height = height
		self.width = width
		self.sprite = sprite
		self.bullet_append = bullet_append_func
		self.direction = Vector2(UP)
		self.screen = screen # the screen where everything gets drawn on
		self.name = name # name is equivalent to an player ID
		self.acceleration = 0.9
		self.draw_position = Vector2() # position where the spaceship gets drawn

	def action(self, action: EnumAction) -> None:
		"""public method to move the ship in the direction of the action"""
		match action:
			case EnumAction.FORWARD:
				new_position_x = self.x + self.direction.x * self.acceleration
				new_position_y = self.y + self.direction.y * self.acceleration
				# correct the position at the end of an action to stay within the screen bounds
				valid_pos_x = np.clip(new_position_x, 0, self.screen.get_width())
				valid_pos_y = np.clip(new_position_y, 0, self.screen.get_height())

				self.x = valid_pos_x
				self.y = valid_pos_y

			case EnumAction.BACKWARD:
				new_position_x = self.x - self.direction.x * self.acceleration
				new_position_y = self.y - self.direction.y * self.acceleration
				# correct the position at the end of an action to stay within the screen bounds
				valid_pos_x = np.clip(new_position_x, 0, self.screen.get_width())
				valid_pos_y = np.clip(new_position_y, 0, self.screen.get_height())

				self.x = valid_pos_x
				self.y = valid_pos_y

			case EnumAction.RIGHT:
				self._rotate(clockwise=True)

			case EnumAction.LEFT:
				self._rotate(clockwise=False)

			case EnumAction.SHOOT:
				self._shoot()

	def _shoot(self) -> None:
		"""public method to create a bullet and append it to the bullet list"""
		bullet_velocity = self.direction * 3
		bullet = Bullet(self.x, self.y - np.floor(self.height/2), self.height, self.width, \
						load_sprite("ai_wars/img/bullet.png"), bullet_velocity)
		self.bullet_append(bullet)

	def _rotate(self, clockwise: bool) -> None:
		"""public method to rotate the ship in clockwise direction"""
		# rotate the direction vector
		sign = 1 if clockwise else -1
		angle = 3 * sign
		self.direction.rotate_ip(angle)

	def draw(self, screen: pygame.Surface) -> None:
		"""public method to draw the rotated version of the spaceship"""
		angle = self.direction.angle_to(UP)
		rotated_surface = pygame.transform.rotozoom(self.sprite, angle, 1.0)
		rotated_surface_size = Vector2(rotated_surface.get_size())
		blit_position = Vector2(self.x, self.y) - rotated_surface_size * 0.5
		screen.blit(rotated_surface, blit_position)

	def update(self, subject: Subject) -> None:
		"""Receive update from subject."""
		pass
