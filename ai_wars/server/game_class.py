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

from ..utils import get_random_position, load_sprite
from ..constants import (
	POINTS_LOST_AFTER_GETTING_HIT,
	POINTS_GAINED_AFTER_HITTING,
	DECREASE_SCORE_EVENT,
	SERVER_TIMEOUT,
	HITSCAN_ENABLED,
	POINTS_LOST_PER_SECOND,
	WIDTH,
	HEIGHT
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
		self.spaceships: Dict[Spaceship] = {}  # dict with every spaceship in the game

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
					self.spawn_spaceship(spawn.x, spawn.y, name)

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
		"""private method to process game logic loop over every bullet and update its position"""
		if HITSCAN_ENABLED:
			bullet_ray_size = 100
			all_hitscan_rays = []
			for spaceship in self.spaceships.values():
				for bullet in self.bullets:
					hitscan_ray = [bullet]
					# if a spaceship has shot a bullet, create a ray of bullets that act as a hitscan mechanism
					bullet_position = Vector2(spaceship.x, spaceship.y)
					for _ in range(bullet_ray_size):
						# in the direction of the shot bullet spawn bullet_ray_size as many bullets
						bullet_position = bullet_position + spaceship.direction.normalize() * 10
						hitscan_ray.append(Bullet(bullet_position.x, bullet_position.y, self.bullet_image,
															spaceship.direction, spaceship))
					# append the ray to the overall ray list. Every entry in the overall ray list is
					# a list of bullets that make up that ray
					all_hitscan_rays.append(hitscan_ray)

			# iterator over every ray and in every ray over every bullet of that ray. Check if any bullet
			# has hit any (except own) spaceship. If so deal the points and remove the whole ray.
			for hitscan_ray in all_hitscan_rays:
				for bullet in hitscan_ray:
					for spaceship in self.spaceships.values():
						if spaceship.hitbox.colliderect(bullet.hitbox):
							# check if bullet hit the shooter of the bullet itself
							if bullet.shooter == spaceship:
								continue
							# remove points from ship that got hit
							shooter_name = bullet.shooter.name
							shot_name = spaceship.name
							self.scoreboard.decrease_score(shot_name, POINTS_LOST_AFTER_GETTING_HIT)
							self.scoreboard.increase_score(shooter_name, POINTS_GAINED_AFTER_HITTING)

							# since a bullet of the ray has hit a other ship, skip the whole ray.
							break
					else:
						continue
					break
				else:
					continue
				break
			# since every shot bullet "turn into" a ray, delete all spawned bullets
			for bullet in self.bullets:
				self.delete_bullet(bullet)

		else:
			for bullet in self.bullets:
				bullet.move(delta_time)
				#If bullet get out of bound then delete it
				if bullet.x > self.screen.get_width() or \
				bullet.x < 0 or \
				bullet.y > self.screen.get_height() or \
				bullet.y < 0:
					self.delete_bullet(bullet)

			# check for collisions of ships and bullets
			# self.scoreboard.decrease_score(ship.name, 100)
			# check if any ships are hit by any bullets
			for ship in self.spaceships.values():
				for bullet in self.bullets:
					if ship.hitbox.colliderect(bullet.hitbox):
						# check if bullet hit the shooter of the bullet itself
						if bullet.shooter == ship:
							continue
						# destroy bullet
						self.delete_bullet(bullet)
						# remove points from ship that got hit
						shooter_name = bullet.shooter.name
						shot_name = ship.name
						self.scoreboard.decrease_score(
							shot_name, POINTS_LOST_AFTER_GETTING_HIT)
						self.scoreboard.increase_score(
							shooter_name, POINTS_GAINED_AFTER_HITTING)


	def delete_bullet(self, bullet) -> None:
		self.bullets.remove(bullet)
		del bullet


	def spawn_spaceship(self, x: int, y: int, name: str) -> None:
		"""spawn a spaceship at the given position"""
		color = [255,0,0]
		spaceship = Spaceship(x, y, self.spaceship_image, self.bullet_image, self.bullets.append, \
							  self.screen, name, color, self.modus.get_game_time())
		self.spaceships[spaceship.name] = spaceship
		self.scoreboard.attach(spaceship)
		logging.debug("Spawned spaceship with name: %s at X:%s Y:%s", name, x, y)


	def _publish_gamestate(self) -> None:
		"""private method to send the current gamestate to all clients"""
		serialized_state = serialize_game_state(self.spaceships.values(), self.bullets,
											   self.scoreboard.get_scoreboard_dict())
		self.server.send_to_all(serialized_state.encode())
