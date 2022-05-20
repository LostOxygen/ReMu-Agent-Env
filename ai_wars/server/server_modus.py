import logging
import abc
import signal
import threading
import pygame

from ..constants import SERVER_TICK_RATE
from ..utils import override


class ServerModus(abc.ABC):

	@abc.abstractmethod
	def start(self, game):
		'''
		Once called when starting the game server.

		Parameters:
			game: the game instance
		'''

		pass

	@abc.abstractmethod
	def received_input(self):
		'''
		Called everytime a new input is received from a client.
		'''

		pass

class Realtime:
	'''
	Runs the game in real time.
	'''

	def __init__(self):
		self.clock = pygame.time.Clock()

	@override
	def start(self, game):
		self.game = game

		# Register handler for SIGINT (Ctrl-C) interrupt
		signal.signal(signal.SIGINT, self.thread_handler)

		# Register timer for decrease score event
		pygame.time.set_timer(self.game.decrease_score_event, 1000)

		# start the update loop of the game
		threading.Thread(target=self.update_loop).start()
		logging.debug("game tick thread started")

	def update_loop(self) -> None:
		"""main loop for input handling, game logic and rendering"""
		# loop over the game loop
		while self.game.running:
			delta_time = self.clock.tick(SERVER_TICK_RATE) / 1000
			self.game.update_game(delta_time)

	def thread_handler(self, _signum, _frame):
		self.game.running = False
		logging.debug("Stopping server threads")

	@override
	def received_input(self):
		pass

class TrainingMode:
	'''
	Runs the game as fast as the slowest client. Main purpose of this mode is for training models.
	'''

	@override
	def start(self, game):
		self.game = game
		self.frame = 0

	@override
	def received_input(self):
		self.frame += 1

		# fire decrease score event if one game second has passed
		if (self.frame * SERVER_TICK_RATE) % 1000 < SERVER_TICK_RATE:
			pygame.event.post(self.game.decrease_score_event)

		# update game state if all clients submitted their action
		if self.game.spaceships.keys() == self.game.action_buffer.keys():
			self.game.update_game(1/SERVER_TICK_RATE)
