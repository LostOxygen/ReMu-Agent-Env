"""Main GameClass"""
import sys
import pygame
from pygame import Vector2
from typing import List, Dict
import threading

from ..spaceship import Spaceship
from ..enums import EnumAction
from ..scoreboard import Scoreboard
from ..bullet import Bullet

from ..networking.client import Client
from .serializer import serialize_action
from .deserializer import deserialize_game_state

from ..utils import load_sprite

POLL_RATE = 30

class GameClass:
	"""MainGameClass"""

	def __init__(self, player_name: str):

		pygame.init()
		self.clock = pygame.time.Clock()
		self.screen = pygame.display.set_mode((800, 600))
		self.background = load_sprite("ai_wars/img/space.png", False)
		# initialize the scoreboard and attach all players as observers
		self.scoreboard = Scoreboard()
		self.bullets: List[Bullet] = []  # list with all bullets in the game
		self.spaceships: Dict[str, Spaceship] = {}  # dict with every spaceship in the game
		self.player_name = player_name

		# initialize server connection
		self.client = Client.builder().with_buffer_size(10*1024).build()


	def main_loop(self) -> None:
		"""main loop for input handling, game logic and rendering"""
		# connect the client to the server
		self.client.connect(addr="127.0.0.1", port=1337)

		# start the client data thread to receive packages from server
		self.client_thread = threading.Thread(target=self.receive_data)
		self.client_thread.start()

		while True:
			self.clock.tick(POLL_RATE)
			self._handle_inputs()
			self._handle_events()


	def receive_data(self) -> None:
		"""data loop to listen and receive data from the server"""
		while True:
			# receive data from server
			data = self.client.recv_next()
			players, projectiles, scoreboard = deserialize_game_state(data.decode())
			print(players, projectiles, scoreboard)
			self._update_players(players)
			self._update_scoreboard(scoreboard)
			self._update_bullets(projectiles)
			self._draw()

	def _handle_inputs(self) -> None:
		"""private method to process inputs and limit the bullet frequency"""
		# action list for all actions of the current tick
		actions: List[EnumAction] = []
		# check which keys are pressed
		is_key_pressed = pygame.key.get_pressed()

		match is_key_pressed:
			case is_key_pressed if is_key_pressed[pygame.K_SPACE]:
				actions.append(EnumAction.SHOOT)
			case is_key_pressed if is_key_pressed[pygame.K_LEFT]:
				actions.append(EnumAction.LEFT)
			case is_key_pressed if is_key_pressed[pygame.K_RIGHT]:
				actions.append(EnumAction.RIGHT)
			case is_key_pressed if is_key_pressed[pygame.K_UP]:
				actions.append(EnumAction.FORWARD)
			case is_key_pressed if is_key_pressed[pygame.K_DOWN]:
				actions.append(EnumAction.BACKWARD)

		actions = serialize_action(self.player_name, actions)
		self.client.send(actions.encode())

	def _handle_events(self) -> None:
		"""Private helper method to listen and process pygame events"""
		for event in pygame.event.get():
			if event.type == pygame.QUIT or \
			   (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
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
				self.spawn_spaceship(player["position"], player["direction"], player_name)


	def _update_scoreboard(self, new_scoreboard: dict) -> None:
		"""private method to newly draw player or update existing players"""
		self.scoreboard.set_scoreboard_dict(new_scoreboard)


	def _update_bullets(self, bullets: list) -> None:
		"""private method to newly draw player or update existing players"""
		self.bullets.clear()

		# iterate over all new bullets and spawn them
		for bullet in bullets:
			self.spawn_bullet(bullet["position"], bullet["direction"], bullet["owner"])


	def _draw(self) -> None:
		"""private method to draw the game"""

		# draw the background
		self.screen.blit(self.background, (0, 0))

		# draw the spaceship
		for spaceship in self.spaceships.values():
			spaceship.draw(self.screen)

		# rendering loop to draw all bullets
		for bullet in self.bullets:
			bullet.draw(self.screen)

		# draw scoreboard
		self.scoreboard.draw_scoreboard(self.screen)

		pygame.display.flip()


	def spawn_spaceship(self, position: Vector2, direction: Vector2, name: str) -> None:
		sprite = load_sprite("ai_wars/img/spaceship.png")
		spaceship = Spaceship(position.x, position.y, sprite, self.bullets.append, self.screen, name)
		spaceship.direction = direction

		self.spaceships[spaceship.name] = spaceship
		self.scoreboard.attach(spaceship)


	def spawn_bullet(self, position: Vector2, direction: Vector2, shooter: str) -> None:
		sprite = load_sprite("ai_wars/img/bullet.png")
		bullet = Bullet(position.x, position.y, sprite, self.screen, shooter)
		bullet.direction = direction

		self.bullets.append(bullet)
