"""spacehip class file"""
from typing import Callable
import pygame
from pygame.math import Vector2
import numpy as np

from ai_wars.enums import EnumAction
from ai_wars.bullet import Bullet
from ai_wars.utils import load_sprite, clip_pos

UP = Vector2(0, -1)


class Spaceship():
	"""spaceship class with functions for moving, drawing and shooting"""
	# constants
	SHOOT_COOLDOWN = 200 # specifies the cooldown for shooting in ms
	MOVEMENT_MULTIPLIER = 2.0
	ROTATION_MULTIPLIER = 3.0

	def __init__(self, x: int, y: int, sprite: pygame.sprite.Sprite, \
				 bullet_append_func: Callable[[Bullet], None], \
				 screen: pygame.Surface, name: str):
		self.x = x
		self.y = y
		self.sprite = sprite
		self.height = sprite.get_rect().height
		self.width = sprite.get_rect().width
		self.bullet_append = bullet_append_func
		self.direction = Vector2(UP)
		self.screen = screen # the screen where everything gets drawn on
		self.name = name # name is equivalent to an player ID

		# hitbox stuff
		self.hitbox = self.sprite.get_rect()
		self.refresh_hitbox_coordinates()

		# bullet cooldown stuff
		self.last_action_time = 0

	def action(self, action: EnumAction) -> None:
		"""public method to move the ship in the direction of the action"""
		match action:
			case EnumAction.FORWARD:
				new_position_x = self.x + self.direction.x * self.MOVEMENT_MULTIPLIER
				new_position_y = self.y + self.direction.y * self.MOVEMENT_MULTIPLIER
				# correct the position at the end of an action to stay within the screen bounds
				valid_pos_x = clip_pos(new_position_x, 0, self.screen.get_width())
				valid_pos_y = clip_pos(new_position_y, 0, self.screen.get_height())

				# move sprite
				self.x = valid_pos_x
				self.y = valid_pos_y

				self.refresh_hitbox_coordinates()

			case EnumAction.BACKWARD:
				new_position_x = self.x - self.direction.x * self.MOVEMENT_MULTIPLIER
				new_position_y = self.y - self.direction.y * self.MOVEMENT_MULTIPLIER
				# correct the position at the end of an action to stay within the screen bounds
				valid_pos_x = clip_pos(new_position_x, 0, self.screen.get_width())
				valid_pos_y = clip_pos(new_position_y, 0, self.screen.get_height())

				self.x = valid_pos_x
				self.y = valid_pos_y

				self.refresh_hitbox_coordinates()

			case EnumAction.RIGHT:
				self._rotate(clockwise=True)

			case EnumAction.LEFT:
				self._rotate(clockwise=False)

			case EnumAction.SHOOT:
				# implementation of shooting cooldown, hence limits the bullets that can be shot
				if pygame.time.get_ticks()-self.last_action_time >= self.SHOOT_COOLDOWN:
					self._shoot()
					self.last_action_time = pygame.time.get_ticks()

	def _shoot(self) -> None:
		"""public method to create a bullet and append it to the bullet list"""
		#TODO We are passing here the wrong height and width (that of the ship), the bullet class
		# can get it itself using the img
		bullet = Bullet(self.x, self.y - np.floor(self.height/2), \
						load_sprite("ai_wars/img/bullet.png"), self.direction, self)

		self.bullet_append(bullet)

	def _rotate(self, clockwise: bool) -> None:
		"""public method to rotate the ship in clockwise direction"""
		# rotate the direction vector
		sign = 1 if clockwise else -1
		angle = self.ROTATION_MULTIPLIER * sign
		self.direction.rotate_ip(angle)

	def draw(self, screen: pygame.Surface) -> None:
		"""public method to draw the rotated version of the spaceship"""
		angle = self.direction.angle_to(UP)
		rotated_surface = pygame.transform.rotozoom(self.sprite, angle, 1.0)
		rotated_surface_size = Vector2(rotated_surface.get_size())
		blit_position = Vector2(self.x, self.y) - rotated_surface_size * 0.5
		screen.blit(rotated_surface, blit_position)

		# debug for drawing hitbox
		#pygame.draw.rect(screen, (0,255,0), self.hitbox)

	def refresh_hitbox_coordinates(self) -> None:
		# this is currently only a hotfix. For some reason self.x and self.y are not in the top
		# left corner as it normally in pygame (and e.g. bullet class) but self.x and self.y give
		# the center of the sprite. Therefore assigning the hitbox center the coordinates, the
		# hitbox aligns with the sprite
		self.hitbox.center = self.x, self.y
