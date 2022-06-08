import abc
import pygame

class GameTime(abc.ABC):
	'''
	Provides the current game time.
	'''

	@abc.abstractmethod
	def get_time(self) -> int:
		return 0

class PygameGameTime(GameTime):

	def get_time(self) -> int:
		return pygame.time.get_ticks()

class VirtualGameTime(GameTime):

	def __init__(self):
		self.game_time = 0

	def increase_game_time(self, passed_time: int):
		self.game_time += passed_time

	def get_time(self) -> int:
		return self.game_time
