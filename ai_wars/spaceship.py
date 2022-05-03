"""spacehip class file"""
import numpy as np
from typing import Callable
import pygame
from pygame.math import Vector2

from ai_wars.enums import EnumAction
from ai_wars.bullet import Bullet
from ai_wars.utils import load_sprite, clip_pos

UP = Vector2(0, -1)


class Spaceship():
	"""spaceship class with functions for moving, drawing and shooting"""
	# constants
	SHOOT_COOLDOWN = 200 # specifies the cooldown for shooting in ms
	MOVEMENT_MULTIPLIER = 200.0
	ROTATION_MULTIPLIER = 300.0

	def __init__(self, x: int, y: int, spaceship_sprite: pygame.sprite.Sprite, bullet_sprite:
				 pygame.sprite.Sprite, bullet_append_func: Callable[[Bullet], None], \
				 screen: pygame.Surface, name: str):
		self.x = x
		self.y = y
		self.spaceship_sprite = spaceship_sprite
		self.bullet_sprite = bullet_sprite
		self.height = spaceship_sprite.get_rect().height
		self.width = spaceship_sprite.get_rect().width
		self.bullet_append = bullet_append_func
		self.direction = Vector2(UP)
		self.screen = screen # the screen where everything gets drawn on
		self.name = name # name is equivalent to an player ID

		# hitbox stuff
		self.hitbox = self.spaceship_sprite.get_rect()
		self.refresh_hitbox_coordinates()

		# bullet cooldown stuff
		self.last_action_time = 0

		# initialize font stuff
		pygame.font.init()
		self.font = pygame.font.SysFont("consolas", 15)
		self.font_width = self.font.size("X")[0]
		self.font_height = self.font.size("X")[1]

		# generate a random color and fill the spaceship with it
		self.color = list(np.random.choice(range(256), size=3))
		spaceship_sprite.fill(self.color, special_flags=pygame.BLEND_MIN)

	def action(self, action: EnumAction, delta_time : float) -> None:
		"""public method to move the ship in the direction of the action"""
		match action:
			case EnumAction.FORWARD:
				new_position_x = self.x + self.direction.x * self.MOVEMENT_MULTIPLIER * delta_time
				new_position_y = self.y + self.direction.y * self.MOVEMENT_MULTIPLIER * delta_time
				# correct the position at the end of an action to stay within the screen bounds
				valid_pos_x = clip_pos(new_position_x, 0, self.screen.get_width())
				valid_pos_y = clip_pos(new_position_y, 0, self.screen.get_height())

				# move sprite
				self.x = valid_pos_x
				self.y = valid_pos_y

				self.refresh_hitbox_coordinates()

			case EnumAction.BACKWARD:
				new_position_x = self.x - self.direction.x * self.MOVEMENT_MULTIPLIER * delta_time
				new_position_y = self.y - self.direction.y * self.MOVEMENT_MULTIPLIER * delta_time
				# correct the position at the end of an action to stay within the screen bounds
				valid_pos_x = clip_pos(new_position_x, 0, self.screen.get_width())
				valid_pos_y = clip_pos(new_position_y, 0, self.screen.get_height())

				self.x = valid_pos_x
				self.y = valid_pos_y

				self.refresh_hitbox_coordinates()

			case EnumAction.RIGHT:
				self._rotate(True, delta_time)

			case EnumAction.LEFT:
				self._rotate(False, delta_time)

			case EnumAction.SHOOT:
				# implementation of shooting cooldown, hence limits the bullets that can be shot
				if pygame.time.get_ticks()-self.last_action_time >= self.SHOOT_COOLDOWN:
					self._shoot()
					self.last_action_time = pygame.time.get_ticks()

	def _shoot(self) -> None:
		"""public method to create a bullet and append it to the bullet list"""
		#TODO Delta time is only given once here, is that ok?
		bullet = Bullet(self.x, self.y, self.bullet_sprite, self.direction, self)
		self.bullet_append(bullet)

	def _rotate(self, clockwise: bool, delta_time: float) -> None:
		"""public method to rotate the ship in clockwise direction"""
		# rotate the direction vector
		sign = 1 if clockwise else -1
		angle = self.ROTATION_MULTIPLIER * sign * delta_time
		self.direction.rotate_ip(angle)

	def draw(self, screen: pygame.Surface) -> None:
		"""public method to draw the rotated version of the spaceship"""

		# draw the spaceship
		angle = self.direction.angle_to(UP)
		rotated_surface = pygame.transform.rotozoom(self.spaceship_sprite, angle, 1.0)
		rotated_surface_size = Vector2(rotated_surface.get_size())
		blit_position = Vector2(self.x, self.y) - rotated_surface_size * 0.5
		screen.blit(rotated_surface, blit_position)

		# draw the players name on top of the spaceshop
		text_surface = self.font.render(self.name, False, (255, 255, 255))
		half_name_length = len(self.name) // 2
		screen.blit(text_surface, (self.x - half_name_length * self.font_width,
								   self.y-self.font_height*3))

	def refresh_hitbox_coordinates(self) -> None:
		# this is currently only a hotfix. For some reason self.x and self.y are not in the top
		# left corner as it normally in pygame (and e.g. bullet class) but self.x and self.y give
		# the center of the sprite. Therefore assigning the hitbox center the coordinates, the
		# hitbox aligns with the sprite
		self.hitbox.center = self.x, self.y
