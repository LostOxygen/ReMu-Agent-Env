"""Bullet class file"""
import pygame
from pygame.math import Vector2


class Bullet():
	"""Bullet class with functions for moving and drawing"""
	def __init__(self, x: int, y: int, height: int, width: int, \
				 sprite: pygame.Surface, velocity: Vector2, shooter):
		self.x = x
		self.y = y
		self.height = height
		self.width = width
		self.sprite = sprite
		self.velocity = velocity
		#Hitbox
		self.hitbox = self.sprite.get_rect()
		self.refresh_hitbox_coordinates()
		#Shooter of the bullet
		self.shooter = shooter


	def move(self) -> None:
		"""public method to move the bullet"""
		#First move sprite
		self.x = self.x + self.velocity.x
		self.y = self.y + self.velocity.y

		self.refresh_hitbox_coordinates()

	def draw(self, surface: pygame.Surface) -> None:
		surface.blit(self.sprite, (self.x, self.y))

		#Debugging - Draw Hitbox
		#pygame.draw.rect(surface, (0,255,0), self.hitbox)

	def refresh_hitbox_coordinates(self):
		self.hitbox.x = self.x
		self.hitbox.y = self.y
