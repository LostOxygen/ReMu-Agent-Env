"""Main GameClass"""
import sys
import os
import pygame
import threading
from typing import List, Dict

from ..spaceship import Spaceship
from ..scoreboard import Scoreboard
from ..bullet import Bullet

from ..networking.server import Server
from .serializer import serialize_game_state
from .deserializer import deserialize_action

from ..utils import get_random_position, load_sprite


class GameClass:
	"""MainGameClass"""

	# constants
	FRAMERATE = 30
	POINTS_LOST_AFTER_GETTING_HIT = 100
	POINTS_GAINED_AFTER_HITTING = 200
	# pygame userevents use codes from 24 to 35, so the first user event will be 24
	DECREASE_SCORE_EVENT = pygame.USEREVENT + 0  # event code 24

	def __init__(self):
		os.putenv("SDL_VIDEODRIVER", "dummy") # start pygame in headless mode
		pygame.init()
		self.clock = pygame.time.Clock()
		self.screen = pygame.display.set_mode((800, 600))
		self.background = load_sprite("ai_wars/img/space.png", False)
		# initialize the scoreboard and attach all players as observers
		self.scoreboard = Scoreboard()
		self.bullets: List[Bullet] = []  # list with all bullets in the game
		self.spaceships: Dict[Spaceship] = {}  # dict with every spaceship in the game

		# initialize custom event timer
		self.decrease_score_event = pygame.event.Event(self.DECREASE_SCORE_EVENT,
												 message="decrease score")
		pygame.time.set_timer(self.decrease_score_event, 1000)

		# initialize server
		self.server = Server.builder().build()
		self.action_buffer = {}


	def main_loop(self) -> None:
		"""main loop for input handling, game logic and rendering"""
		# start the server thread to receive packages
		self.server_thread = threading.Thread(target=self.server_loop)
		self.server_thread.start()

		# loop over the game loop
		while True:
			self.clock.tick(self.FRAMERATE)
			self._handle_events()
			self._process_game_logic()
			self._apply_actions()
			self._publish_gamestate()


	def server_loop(self) -> None:
		"""server loop to handle receive modularity"""
		self.server.start(addr="127.0.0.1", port=1337)
		while True:
			received_action = self.server.recv_next()
			name, actions = deserialize_action(received_action)

			# spawn spaceship at random position if necessary
			if name not in self.spaceships:
				spawn = get_random_position(self.screen)
				self.spawn_spaceship(spawn.x, spawn.y, name)

			# store actions in buffer
			if name not in self.action_buffer:
				self.action_buffer[name] = set()
			self.action_buffer[name].update(actions)


	def _handle_events(self) -> None:
		"""Private helper method to listen and process pygame events"""
		for event in pygame.event.get():
			match event.type:
				# check if the game should be closed
				case pygame.QUIT:
					sys.exit()

				# decrease the score of the players (event gets fired every second)
				case self.DECREASE_SCORE_EVENT:
					for ship in self.spaceships.values():
						self.scoreboard.decrease_score(ship.name, 1)

	def _apply_actions(self):
		'''private method to applies all actions in the action buffer, then clears it'''
		for (name, actions) in self.action_buffer.items():
			for action in actions:
				self.spaceships[name].action(action)

		self.action_buffer.clear()


	def _process_game_logic(self) -> None:
		"""private method to process game logic"""
		# loop over every bullet and update its position
		for bullet in self.bullets:
			bullet.move()
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
						shot_name, self.POINTS_LOST_AFTER_GETTING_HIT)
					self.scoreboard.increase_score(
						shooter_name, self.POINTS_GAINED_AFTER_HITTING)


	def delete_bullet(self, bullet) -> None:
		self.bullets.remove(bullet)
		del bullet


	def spawn_spaceship(self, x: int, y: int, name: str) -> None:
		"""spawn a spaceship at the given position"""
		sprite = load_sprite("ai_wars/img/spaceship.png")
		spaceship = Spaceship(x, y, sprite, self.bullets.append, self.screen, name)
		self.spaceships[spaceship.name] = spaceship
		self.scoreboard.attach(spaceship)


	def _publish_gamestate(self) -> None:
		"""private method to send the current gamestate to all clients"""
		serialized_state = serialize_game_state(self.spaceships.values(), self.bullets,
											   self.scoreboard.get_scoreboard_dict())
		self.server.send_to_all(serialized_state.encode())
