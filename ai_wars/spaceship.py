"""spacehip class file"""
import pygame
from pygame.math import Vector2

from ai_wars.enums import EnumAction
from ai_wars.game_time import GameTime, PygameGameTime
from ai_wars.utils import clip_pos
from ai_wars.constants import (
	SHIP_SPEED,
	ROTATION_SPEED
)


UP = Vector2(0, -1)


class Spaceship():
	"""spaceship class with functions for moving, drawing and shooting"""

	def __init__(self,
		x: int,
		y: int,
		spaceship_sprite: pygame.sprite.Sprite,
		screen: pygame.Surface,
		name: str,
		color: list,
		direction: Vector2,
		game_time: GameTime = PygameGameTime()
	):
		self.x = x
		self.y = y
		self.spaceship_sprite = spaceship_sprite.copy()
		self.height = spaceship_sprite.get_rect().height
		self.width = spaceship_sprite.get_rect().width
		self.direction = direction.copy()
		self.screen = screen # the screen where everything gets drawn on
		self.name = name # name is equivalent to an player ID
		self.color = color
		self.game_time = game_time

		# hitbox stuff
		self.hitbox = self.spaceship_sprite.get_rect()
		self.refresh_hitbox_coordinates()

		# initialize font stuff
		pygame.font.init()
		self.font = pygame.font.SysFont("consolas", 15)
		self.font_width = self.font.size("X")[0]
		self.font_height = self.font.size("X")[1]

		self.spaceship_sprite.fill(self.color, special_flags=pygame.BLEND_MIN)

	def action(self, action: EnumAction, delta_time : float) -> None:
		"""public method to move the ship in the direction of the action"""
		match action:
			case EnumAction.FORWARD:
				new_position_x = self.x + self.direction.x * SHIP_SPEED * delta_time
				new_position_y = self.y + self.direction.y * SHIP_SPEED * delta_time
				# correct the position at the end of an action to stay within the screen bounds
				valid_pos_x = clip_pos(new_position_x, 0, self.screen.get_width())
				valid_pos_y = clip_pos(new_position_y, 0, self.screen.get_height())

				# move sprite
				self.x = valid_pos_x
				self.y = valid_pos_y

				self.refresh_hitbox_coordinates()

			case EnumAction.BACKWARD:
				new_position_x = self.x - self.direction.x * SHIP_SPEED * delta_time
				new_position_y = self.y - self.direction.y * SHIP_SPEED * delta_time
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

	def _rotate(self, clockwise: bool, delta_time: float) -> None:
		"""public method to rotate the ship in clockwise direction"""
		# rotate the direction vector
		sign = 1 if clockwise else -1
		angle = ROTATION_SPEED * sign * delta_time
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
