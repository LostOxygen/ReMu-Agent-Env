"""Main GameClass"""
import sys
import logging
import pygame
from pygame import Vector2
from typing import List, Dict, Set

from .behavior import Behavior

from ..spaceship import Spaceship
from ..enums import EnumAction
from ..scoreboard import Scoreboard
from ..bullet import Bullet

from ..utils import load_sprite, override


class GameGUI(Behavior):
	"""Simple game GUI with user inputs representing a player behavior"""

	def __init__(self):
		self.clock = pygame.time.Clock()
		self.screen = pygame.display.set_mode((800, 600))
		self.background = load_sprite("ai_wars/img/Background.png", False)

		# data structures that hold the game information
		self.scoreboard = Scoreboard()
		self.bullets: List[Bullet] = []  # list with all bullets in the game
		self.spaceships: Dict[str, Spaceship] = {}  # dict with every spaceship in the game

		pygame.init()
		logging.debug("Initialized client")


	@override
	def make_move(self,
		players: dict[str, any],
		projectiles: dict[str, any],
		scoreboard: dict[str, int]
	) -> set[EnumAction]:
		self._handle_events()

		self._update_players(players)
		self._update_scoreboard(scoreboard)
		self._update_bullets(projectiles)
		self._draw()

		return self._handle_inputs()


	def _handle_inputs(self) -> None:
		"""private method to process inputs and limit the bullet frequency"""
		# action list for all actions of the current tick
		actions: Set[EnumAction] = set()
		# check which keys are pressed
		is_key_pressed = pygame.key.get_pressed()

		if is_key_pressed[pygame.K_SPACE]:
			actions.add(EnumAction.SHOOT)
		if is_key_pressed[pygame.K_LEFT]:
			actions.add(EnumAction.LEFT)
		if is_key_pressed[pygame.K_RIGHT]:
			actions.add(EnumAction.RIGHT)
		if is_key_pressed[pygame.K_UP]:
			actions.add(EnumAction.FORWARD)
		if is_key_pressed[pygame.K_DOWN]:
			actions.add(EnumAction.BACKWARD)

		return actions

	def _handle_events(self) -> None:
		"""Private helper method to listen and process pygame events"""
		for event in pygame.event.get():
			if event.type == pygame.QUIT or \
			   (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
				logging.debug("Received quit event")
				sys.exit()


	def _update_players(self, players: list) -> None:
		"""private method to newly draw player or update existing players"""
		for player in players:
			# check if player already exists
			player_name = player["player_name"]
			if player_name in self.spaceships:
				# update player
				self.spaceships[player_name].x = player["position"].x
				self.spaceships[player_name].y = player["position"].y
				self.spaceships[player_name].direction = player["direction"]
			else:
				# create new player
				self._spawn_spaceship(player["position"], player["direction"], player_name)


	def _update_scoreboard(self, new_scoreboard: dict) -> None:
		"""private method to newly draw player or update existing players"""
		self.scoreboard.set_scoreboard_dict(new_scoreboard)


	def _update_bullets(self, bullets: list) -> None:
		"""private method to newly draw player or update existing players"""
		self.bullets.clear()

		# iterate over all new bullets and spawn them
		for bullet in bullets:
			self._spawn_bullet(bullet["position"], bullet["direction"], bullet["owner"])


	def _draw(self) -> None:
		"""private method to draw the game"""

		# draw the background
		self.screen.blit(self.background, (0, 0))

		# rendering loop to draw all bullets
		for bullet in self.bullets:
			bullet.draw(self.screen)

		# draw the spaceship
		for spaceship in self.spaceships.values():
			spaceship.draw(self.screen)

		# draw scoreboard
		self.scoreboard.draw_scoreboard(self.screen)

		pygame.display.flip()


	def _spawn_spaceship(self, position: Vector2, direction: Vector2, name: str) -> None:
		sprite = load_sprite("ai_wars/img/spaceship.png")
		spaceship = Spaceship(position.x, position.y, sprite, self.bullets.append, self.screen, name)
		spaceship.direction = direction

		self.spaceships[spaceship.name] = spaceship
		self.scoreboard.attach(spaceship)


	def _spawn_bullet(self, position: Vector2, direction: Vector2, shooter: str) -> None:
		sprite = load_sprite("ai_wars/img/bullet.png")
		bullet = Bullet(position.x, position.y, sprite, direction, shooter)

		self.bullets.append(bullet)
