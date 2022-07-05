"""Main GameClass"""
import os
import sys
from datetime import datetime
from copy import copy

import pygame
from pygame.math import Vector2
import logging
from typing import Dict

from .server_modus import ServerModus

from ..spaceship import Spaceship
from ..maps.map import Checkpoint
from ..scoreboard import Scoreboard

from ..networking.server import UdpServer
from ..networking.layers.compression import GzipCompression
from .serializer import serialize_game_state
from .deserializer import deserialize_action
from ..maps.map_loader import load_map

from ..utils import load_sprite
from ..constants import (
	SERVER_TIMEOUT,
	WIDTH,
	HEIGHT,
	MAP,
	POINT_PER_CHECKPOINT
)


class GameClass:
	"""MainGameClass"""
	# images
	os.putenv("SDL_VIDEODRIVER", "dummy")  # start pygame in headless mode
	screen = pygame.display.set_mode((WIDTH, HEIGHT))
	spaceship_image = load_sprite("ai_wars/img/spaceship.png", True)

	def __init__(self, modus: ServerModus, addr: str, port: int):
		pygame.init()

		self.modus = modus

		# initialize the scoreboard and attach all players as observers
		self.scoreboard = Scoreboard()
		self.spaceships: Dict[str, Spaceship] = {}  # dict with every spaceship in the game

		logging.debug("Initialized server")

		# initialize server
		self.addr = addr
		self.port = port
		self.server = UdpServer.builder() \
			.with_timeout(SERVER_TIMEOUT) \
			.add_layer(GzipCompression()) \
			.build()
		self.action_buffer = {}

		self.running = True

		self.map = load_map(self.screen, MAP)
		self.checkpoint_by_player: dict[str, int] = {}

	def update_game(self, deltatime):
		self._handle_events()
		self._process_game_logic()
		self._apply_actions(deltatime)
		self._publish_gamestate()

		for ship in self.spaceships.keys():
			self.log_metrics(ship)

	def loop(self) -> None:
		"""server loop to handle receive modularity"""
		self.server.start(addr=self.addr, port=self.port)

		self.modus.start(self)

		while self.running:
			hit_timeout = False
			try:
				received_action = self.server.recv_next()
				name, actions = deserialize_action(received_action)

				if name == "spectator":
					continue

				# spawn spaceship at random position if necessary
				if name not in self.spaceships:
					self.spawn_spaceship(self.map.spawn_point.x, self.map.spawn_point.y, name)

				# store actions in buffer
				if name not in self.action_buffer:
					self.action_buffer[name] = set()
				self.action_buffer[name].update(actions)
			except (TimeoutError, ConnectionResetError):
				hit_timeout = True

			self.modus.received_input(hit_timeout)

	def _handle_events(self) -> None:
		"""Private helper method to listen and process pygame events"""
		for event in pygame.event.get():
			match event.type:
				# check if the game should be closed
				case pygame.QUIT:
					logging.debug("Received Pygame.QUIT event")
					self.thread_handler()
					sys.exit()

	def _apply_actions(self, delta_time):
		'''private method to applies all actions in the action buffer, then clears it'''
		for (name, actions) in self.action_buffer.items():
			for action in actions:
				self.spaceships[name].action(action, delta_time)

		self.action_buffer.clear()

	def _process_game_logic(self) -> None:
		# update scores
		# Check if in bounds or on checkpoint
		for spaceship in self.spaceships.values():
			spaceship_location = Vector2(spaceship.x, spaceship.y)

			# When hit boundary respawn
			if not self.map.is_point_in_bounds(spaceship_location):
				self.respawn_ship(spaceship)
				self.scoreboard.update_score(spaceship.name, -1000000)
				continue

			next_checkpoint_index = self.checkpoint_by_player[spaceship.name]

			# reached next checkpoint
			if self.map.is_point_on_checkpoint(next_checkpoint_index, spaceship_location):
				next_checkpoint_index += 1

			# we passed the last checkpoint (goal), reset
			if next_checkpoint_index >= len(self.map.checkpoints):
				self.respawn_ship(spaceship)
				self.scoreboard.update_score(spaceship.name, 1000000)
				self.scoreboard.increment_finish_reached(spaceship.name)
				continue
			self.checkpoint_by_player[spaceship.name] = next_checkpoint_index

			distance_to_player = self.map.checkpoints[next_checkpoint_index].middle_point \
				.distance_to(spaceship_location)
			distance = next_checkpoint_index * POINT_PER_CHECKPOINT \
				+ (POINT_PER_CHECKPOINT - distance_to_player)
			self.scoreboard.update_score(spaceship.name, distance)

	def delete_bullet(self, bullet) -> None:
		self.bullets.remove(bullet)
		del bullet

	def spawn_spaceship(self, x: int, y: int, name: str) -> None:
		"""spawn a spaceship at the given position"""
		color = [255, 0, 0]
		spaceship = Spaceship(x, y, self.spaceship_image, \
							  self.screen, name, color, self.map.spawn_direction, self.modus.get_game_time())

		self.spaceships[spaceship.name] = spaceship
		self.scoreboard.attach(spaceship)
		self.checkpoint_by_player[spaceship.name] = 0
		logging.debug("Spawned spaceship with name: %s at X:%s Y:%s", name, x, y)

	def _publish_gamestate(self) -> None:
		"""private method to send the current gamestate to all clients"""
		serialized_state = serialize_game_state(self.spaceships.values(),
												self.scoreboard.get_scoreboard_dict())
		self.server.send_to_all(serialized_state.encode())

	def respawn_ship(self, spaceship: Spaceship):
		spaceship.x = self.map.spawn_point.x
		spaceship.y = self.map.spawn_point.y
		spaceship.direction = copy(self.map.spawn_direction)

		self.scoreboard.increase_attempts(spaceship.name)
		self.checkpoint_by_player[spaceship.name] = 0

	def log_metrics(self, name) -> None:
		"""
		Logs attempts and total goals for a model and saves the log under ./logs/model_name_racing.log
		"""

		if not os.path.exists("logs/"):
			os.mkdir("logs/")

		attempts = self.scoreboard.get_scoreboard_dict()[name].attempts
		finish_reached = self.scoreboard.get_scoreboard_dict()[name].finish_reached

		try:
			with open(f"./logs/{name}_racing.log", encoding="utf-8", mode="a") as log_file:
				log_file.write(f"{datetime.now().strftime('%A, %d. %B %Y %I:%M%p')} "
							f"- attempt: {attempts} - total_goals: {finish_reached}\n")
		except OSError as error:
			logging.error("Could not write logs into /logs/%s_racing.log - error: %s",
				self.name, error)

def checkpoint_distance(c1: Checkpoint, c2: Checkpoint) -> float:
	return c1.middle_point.distance_to(c2.middle_point)
