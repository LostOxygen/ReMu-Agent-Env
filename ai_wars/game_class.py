"""Main GameClass"""
import sys
import pygame

from ai_wars.utils import load_sprite
from ai_wars.spaceship import Spaceship
from ai_wars.enums import EnumAction


class GameClass:
	"""MainGameClass"""
	def __init__(self):

		pygame.init()
		self.screen = pygame.display.set_mode((800, 600))
		self.clock = pygame.time.Clock()
		self.background = load_sprite("ai_wars/img/space.png", False)
		self.bullets = []
		self.spaceship = Spaceship(400, 300, 40, 40,
								   load_sprite("ai_wars/img/spaceship.png"),
								   self.bullets.append, self.screen)

	def main_loop(self) -> None:
		"""main loop for input handling, game logic and rendering"""
		while True:
			self._handle_input()
			self._process_game_logic()
			self._draw()

	def _handle_input(self) -> None:
		"""Private helper method to listen for and process input via
			pygame events
		"""
		for event in pygame.event.get():
			if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN \
				and event.key == pygame.K_ESCAPE):
				sys.exit()
			elif (
					self.spaceship
					and event.type == pygame.KEYDOWN
					and event.key == pygame.K_SPACE
			):
				self.spaceship.shoot()

		is_key_pressed = pygame.key.get_pressed()

		if is_key_pressed[pygame.K_UP]:
			self.spaceship.move(EnumAction.FORWARD)

		if is_key_pressed[pygame.K_DOWN]:
			self.spaceship.move(EnumAction.BACKWARDS)

		if is_key_pressed[pygame.K_RIGHT]:
			self.spaceship.move(EnumAction.RIGHT)

		if is_key_pressed[pygame.K_LEFT]:
			self.spaceship.move(EnumAction.LEFT)

	def _draw(self) -> None:
		self.screen.blit(self.background, (0, 0))
		self.spaceship.draw(self.screen)

		# Debugging:
		# print(len(self.bullets))
		for bullet in self.bullets:
			if (bullet.x > self.screen.get_width() or
				bullet.x < 0 or
				bullet.y > self.screen.get_height() or
				bullet.y < 0):
				self.bullets.remove(bullet)
				del bullet
			else:
				bullet.draw(self.screen)

		pygame.display.flip()
		self.clock.tick(75)

	def _process_game_logic(self) -> None:
		for bullet in self.bullets:
			bullet.move()
