import logging
import abc
import signal
import threading
from typing import Any
import pygame

from ..game_time import GameTime, PygameGameTime, VirtualGameTime
from ..constants import SERVER_TICK_RATE
from ..utils import override


class ServerModus(abc.ABC):
	'''
	Abstract server modus class.
	'''

	@abc.abstractmethod
	def start(self, game):
		'''
		Once called when starting the game server.

		Parameters:
			game: the game instance
		'''

		pass

	@abc.abstractmethod
	def received_input(self, hit_timeout: bool):
		'''
		Called every time a new input is received from a client.

		Parameters:
			hit_timeout: whether a socket read timeout was triggered
		'''

		pass

	@abc.abstractmethod
	def get_game_time(self) -> GameTime:
		'''
		Returns an object that provides the current game time.

		Returns:
			game time object
		'''

		pass

class Realtime:
	'''
	Runs the game in real time.
	'''

	def __init__(self):
		self.clock = pygame.time.Clock()
		self.game_time = PygameGameTime()

	@override
	def start(self, game):
		self.game = game

		# Register handler for SIGINT (Ctrl-C) interrupt
		signal.signal(signal.SIGINT, self.thread_handler)

		# Register timer for decrease score event
		pygame.time.set_timer(self.game.decrease_score_event, 1000)

		# start the update loop of the game
		threading.Thread(target=self.update_loop).start()
		logging.debug("Game tick thread started")

	def update_loop(self) -> None:
		"""main loop for input handling, game logic and rendering"""
		# loop over the game loop
		while self.game.running:
			delta_time = self.clock.tick(SERVER_TICK_RATE) / 1000
			self.game.update_game(delta_time)

	def thread_handler(self, signum: Any, frame: Any) -> None:  # pylint: disable=unused-argument
		self.game.running = False
		logging.debug("Stopping server threads")

	@override
	def received_input(self, hit_timeout: bool):
		pass

	@override
	def get_game_time(self) -> GameTime:
		return self.game_time

class TrainingMode:
	'''
	Runs the game as fast as the slowest client. Main purpose of this mode is for training models.
	'''

	def __init__(self):
		self.game_time = VirtualGameTime()

	@override
	def start(self, game):
		self.game = game
		self.frame = 0
		self.last_decrease = 0

	@override
	def received_input(self, hit_timeout: bool):
		self.game_time.increase_game_time(1/SERVER_TICK_RATE * 1000)

		# fire decrease score event if one game second has passed
		time = self.game_time.get_time()
		if time - self.last_decrease > 1000:
			pygame.event.post(self.game.decrease_score_event)
			self.last_decrease = time

		# update game state if all clients submitted their action
		if hit_timeout or self.game.spaceships.keys() == self.game.action_buffer.keys():
			self.frame += 1
			self.game.update_game(1/SERVER_TICK_RATE)

	@override
	def get_game_time(self) -> GameTime:
		return self.game_time
