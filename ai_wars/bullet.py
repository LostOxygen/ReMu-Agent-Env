"""Bullet class file"""
import pygame
from pygame.math import Vector2


class Bullet():
	"""Bullet class with functions for moving and drawing"""
	# constants
	MOVEMENT_MULTIPLIER = 400.0

	def __init__(self, x: int, y: int, sprite: pygame.Surface, direction: Vector2, shooter):
		self.x = x
		self.y = y
		self.sprite = sprite
		self.height = sprite.get_rect().height
		self.width = sprite.get_rect().width
		self.velocity = direction.normalize() * self.MOVEMENT_MULTIPLIER
		# hitbox
		self.hitbox = self.sprite.get_rect()
		self.refresh_hitbox_coordinates()
		# shooter of the bullet
		self.shooter = shooter


	def move(self, delta_time) -> None:
		"""public method to move the bullet"""
		#First move sprite
		self.x = self.x + self.velocity.x * delta_time
		self.y = self.y + self.velocity.y * delta_time

		self.refresh_hitbox_coordinates()

	def draw(self, surface: pygame.Surface) -> None:
		surface.blit(self.sprite, (self.x, self.y))

	def refresh_hitbox_coordinates(self):
		self.hitbox.x = self.x 
		self.hitbox.y = self.y 