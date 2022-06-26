"""Main GameClass"""
import random
import numpy as np
import sys
import logging
import pygame
from pygame import Vector2
from typing import Dict, Set

from .behavior import Behavior

from ..spaceship import Spaceship
from ..enums import EnumAction
from ..scoreboard import Scoreboard
from ..maps.map_loader import load_map

from ..utils import load_sprite, override
from ..constants import (
	COLOR_ARRAY,
	WIDTH,
	HEIGHT,
	MAP
)


class GameGUI(Behavior):
	"""Simple game GUI with user inputs representing a player behavior"""
	# images
	screen = pygame.display.set_mode((WIDTH, HEIGHT))
	background_image = load_sprite("ai_wars/img/background.png", True, False)
	spaceship_image = load_sprite("ai_wars/img/spaceship.png", True, False)

	def __init__(self):
		self.clock = pygame.time.Clock()
		# data structures that hold the game information
		self.scoreboard = Scoreboard()
		self.spaceships: Dict[str, Spaceship] = {}  # dict with every spaceship in the game
		self.color_array = COLOR_ARRAY.copy()
		self.map = load_map(self.screen, MAP)

		pygame.init()
		logging.debug("Initialized client")


	@override
	def make_move(self,
		players: list[dict[str, any]],
		scoreboard: dict[str, int]
	) -> set[EnumAction]:
		self._handle_events()

		self._update_players(players)
		self._update_scoreboard(scoreboard)
		self._draw()

		return self._handle_inputs()


	def _handle_inputs(self) -> None:
		"""private method to process inputs"""
		# action list for all actions of the current tick
		actions: Set[EnumAction] = set()
		# check which keys are pressed
		is_key_pressed = pygame.key.get_pressed()

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


	def _draw(self) -> None:
		"""private method to draw the game"""
		# draw the background
		self.screen.blit(self.background_image, (0, 0))

		self.map.draw()

		# draw the spaceship
		for spaceship in self.spaceships.values():
			spaceship.draw(self.screen)

		# draw scoreboard
		self.scoreboard.draw_scoreboard(self.screen)

		pygame.display.flip()


	def _spawn_spaceship(self, position: Vector2, direction: Vector2, name: str) -> None:
		if self.color_array:
			color = random.choice(self.color_array)
			self.color_array.remove(color)
		else:
			while True:
				color = list(np.random.choice(range(256), size=3))
				if not self._isColorTooDark(color):
					break

		spaceship = Spaceship(position.x, position.y, self.spaceship_image, self.screen, name, color)
		spaceship.direction = direction
		self.spaceships[spaceship.name] = spaceship
		self.scoreboard.attach(spaceship)

	def _isColorTooDark(self, color: list):
		# calc the brightness of the color
		luma = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
		return luma < 128
