"""scoreboard library"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
import pygame


class Subject(ABC):
	"""The Subject interface."""

	@abstractmethod
	def attach(self, observer: Observer) -> None:
		"""Attach an observer to the subject."""
		pass

	@abstractmethod
	def detach(self, observer: Observer) -> None:
		"""Detach an observer from the subject."""
		pass

	@abstractmethod
	def notify(self) -> None:
		"""Notify all observers about an event."""
		pass


class Observer(ABC):
	"""Observer interface."""

	@abstractmethod
	def update(self, subject: Subject) -> None:
		"""Receive update from subject."""
		pass


class Scoreboard(Subject):
	"""The Scoreboard subject notifies observers when the state changes."""

	START_SCORE = 1000 # start score of every player
	_observers: List[Observer] = []
	_scoreboard_dict: Dict[str, int] = {}

	def __init__(self):
		pygame.font.init()
		self.font = pygame.font.SysFont("consolas", 15)
		self.font_width = self.font.size("X")[0]
		self.font_height = self.font.size("X")[1]

	def attach(self, observer: Observer) -> None:
		self._observers.append(observer)
		self._scoreboard_dict[observer.name] = self.START_SCORE

	def detach(self, observer: Observer) -> None:
		self._observers.remove(observer)
		del self._scoreboard_dict[observer.name]

	def notify(self) -> None:
		for observer in self._observers:
			observer.update(self)

	def update_score(self, player_name: str, new_score: int) -> None:
		self._scoreboard_dict[player_name] = new_score
		# re-sort the scoreboard
		self._scoreboard_dict = dict(sorted(self._scoreboard_dict.items(),
									 key=lambda x: x[1], reverse=True))
		self.notify()

	def decrease_score(self, player_name: str, decrease: int) -> None:
		new_score_val = np.clip(self._scoreboard_dict[player_name] - decrease, 0, None)
		self._scoreboard_dict[player_name] = new_score_val
		# re-sort the scoreboard
		self._scoreboard_dict = dict(sorted(self._scoreboard_dict.items(),
                                      key=lambda x: x[1], reverse=True))
		self.notify()

	def increase_score(self, player_name: str, increase: int) -> None:
		new_score_val = np.clip(self._scoreboard_dict[player_name] + increase, 0, None)
		self._scoreboard_dict[player_name] = new_score_val
		# re-sort the scoreboard
		self._scoreboard_dict = dict(sorted(self._scoreboard_dict.items(),
                                      key=lambda x: x[1], reverse=True))
		self.notify()

	def draw_scoreboard(self, screen: pygame.Surface) -> None:
		"""public method to draw the scoreboard on the given screen"""
		for pos, (player, score) in enumerate(self._scoreboard_dict.items()):
			score_string = f"{player} : {score}"
			text_surface = self.font.render(score_string, False, (255, 255, 255))
			screen.blit(text_surface, (0, self.font_height*pos))
