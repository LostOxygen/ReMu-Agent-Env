"""Main GameClass"""
import sys
import os

import pygame
from pygame.math import Vector2
import logging
from typing import List, Dict

from .server_modus import ServerModus

from ..spaceship import Spaceship
from ..scoreboard import Scoreboard
from ..bullet import Bullet

from ..networking.server import UdpServer
from ..networking.layers.compression import GzipCompression
from .serializer import serialize_game_state
from .deserializer import deserialize_action
from ..maps.map_loader import load_map

from ..utils import get_random_position, load_sprite
from ..constants import (
	POINTS_LOST_AFTER_GETTING_HIT,
	POINTS_GAINED_AFTER_HITTING,
	DECREASE_SCORE_EVENT,
	SERVER_TIMEOUT,
	HITSCAN_ENABLED,
	POINTS_LOST_PER_SECOND,
	WIDTH,
	HEIGHT,
	MAP,
	MAX_POINTS_WHEN_GOAL_REACHED,
	MAX_ITERATIONS
)

class GameClass:
	"""MainGameClass"""
	# images
	os.putenv("SDL_VIDEODRIVER", "dummy") # start pygame in headless mode
	screen = pygame.display.set_mode((WIDTH, HEIGHT))
	spaceship_image = load_sprite("ai_wars/img/spaceship.png", True)
	bullet_image = load_sprite("ai_wars/img/bullet.png", True)


	def __init__(self, modus: ServerModus, addr: str, port: int):
		pygame.init()

		self.modus = modus

		# initialize the scoreboard and attach all players as observers
		self.scoreboard = Scoreboard()
		self.bullets: List[Bullet] = []  # list with all bullets in the game
		self.spaceships: Dict[str, Spaceship] = {}  # dict with every spaceship in the game

		# initialize custom event timer
		self.decrease_score_event = pygame.event.Event(DECREASE_SCORE_EVENT,
													   message="decrease score")

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

	def update_game(self, deltatime):
		self._handle_events()
		self._process_game_logic(deltatime)
		self._apply_actions(deltatime)
		self._publish_gamestate()


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
					spawn = get_random_position(self.screen)
					self.spawn_spaceship(self.map.spawn_point.x, self.map.spawn_point.y, name)

				# store actions in buffer
				if name not in self.action_buffer:
					self.action_buffer[name] = set()
				self.action_buffer[name].update(actions)
			except (TimeoutError , ConnectionResetError):
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

				# decrease the score of the players (event gets fired every second)
				case _ if event.type == DECREASE_SCORE_EVENT:
					for ship in self.spaceships.values():
						self.scoreboard.decrease_score(ship.name, POINTS_LOST_PER_SECOND)


	def _apply_actions(self, delta_time):
		'''private method to applies all actions in the action buffer, then clears it'''
		for (name, actions) in self.action_buffer.items():
			for action in actions:
				self.spaceships[name].action(action, delta_time)

		self.action_buffer.clear()


	def _process_game_logic(self, delta_time) -> None:
		for spaceship in self.spaceships.values():
			spaceship_location = Vector2(spaceship.x, spaceship.y)

			current_dist = spaceship_location.distance_squared_to(self.map.goal_point)
			percent_dist = current_dist / self.map.max_dist_between_spawn_and_goal

			self.scoreboard.update_score(spaceship.name, int((1-percent_dist)*MAX_POINTS_WHEN_GOAL_REACHED))

		# Check if in bounds
		for spaceship in self.spaceships.values():
			# When hit goal respawn and give points and increment
			if self.map.goal_rect.collidepoint(Vector2(spaceship.x, spaceship.y)):
				self.respawn_ship(spaceship)
				self.scoreboard.increase_score(spaceship.name, 1000000)
				self.scoreboard.increment_goal_reached(spaceship.name)

			# When it boundary respawn
			if not self.map.is_point_in_bounds(Vector2(spaceship.x, spaceship.y)):
				self.respawn_ship(spaceship)

	def delete_bullet(self, bullet) -> None:
		self.bullets.remove(bullet)
		del bullet


	def spawn_spaceship(self, x: int, y: int, name: str) -> None:
		"""spawn a spaceship at the given position"""
		color = [255,0,0]
		spaceship = Spaceship(x, y, self.spaceship_image, self.bullet_image, self.bullets.append, \
							  self.screen, name, color, self.map.spawn_direction, self.modus.get_game_time())
		self.spaceships[spaceship.name] = spaceship
		self.scoreboard.attach(spaceship)
		logging.debug("Spawned spaceship with name: %s at X:%s Y:%s", name, x, y)


	def _publish_gamestate(self) -> None:
		"""private method to send the current gamestate to all clients"""
		serialized_state = serialize_game_state(self.spaceships.values(), self.bullets,
											   self.scoreboard.get_scoreboard_dict())
		self.server.send_to_all(serialized_state.encode())

	def respawn_ship(self, spaceship: Spaceship):
		spaceship.x = self.map.spawn_point.x
		spaceship.y = self.map.spawn_point.y
		spaceship.direction = self.map.spawn_direction.copy()

