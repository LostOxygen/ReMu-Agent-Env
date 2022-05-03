"""scoreboard library"""
from __future__ import annotations
# from abc import ABC, abstractmethod
from typing import List, Dict

from .utils import clip
import pygame


# class Subject(ABC):
# 	"""The Subject interface."""

# 	@abstractmethod
# 	def attach(self, observer: Observer) -> None:
# 		"""Attach an observer to the subject."""
# 		pass

# 	@abstractmethod
# 	def detach(self, observer: Observer) -> None:
# 		"""Detach an observer from the subject."""
# 		pass

# 	@abstractmethod
# 	def notify(self) -> None:
# 		"""Notify all observers about an event."""
# 		pass


# class Observer(ABC):
# 	"""Observer interface."""

# 	@abstractmethod
# 	def update(self, subject: Subject) -> None:
# 		"""Receive update from subject."""
# 		pass


class Scoreboard():
	"""The Scoreboard subject notifies observers when the state changes."""

	START_SCORE = 1000 # start score of every player
	_observers: List = []
	_scoreboard_dict: Dict[str, int] = {}


	def __init__(self):
		pygame.font.init()
		self.font = pygame.font.SysFont("consolas", 15)
		self.font_width = self.font.size("X")[0]
		self.font_height = self.font.size("X")[1]


	def attach(self, observer) -> None:
		self._observers.append(observer)
		self._scoreboard_dict[observer.name] = self.START_SCORE


	def detach(self, observer) -> None:
		self._observers.remove(observer)
		del self._scoreboard_dict[observer.name]


	def notify(self) -> None:
		pass


	def update_score(self, player_name: str, new_score: int) -> None:
		self._scoreboard_dict[player_name] = new_score
		# re-sort the scoreboard
		self._scoreboard_dict = dict(sorted(self._scoreboard_dict.items(),
									 key=lambda x: x[1], reverse=True))
		self.notify()


	def decrease_score(self, player_name: str, decrease: int) -> None:
		new_score_val = clip(self._scoreboard_dict[player_name] - decrease)
		self._scoreboard_dict[player_name] = new_score_val
		# re-sort the scoreboard
		self._scoreboard_dict = dict(sorted(self._scoreboard_dict.items(),
                                     key=lambda x: x[1], reverse=True))
		self.notify()


	def increase_score(self, player_name: str, increase: int) -> None:
		new_score_val = clip(self._scoreboard_dict[player_name] + increase)
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


	def get_scoreboard_dict(self) -> Dict[str, int]:
		"""public method to get the scoreboard dict"""
		return self._scoreboard_dict


	def set_scoreboard_dict(self, new_scoreboard_dict: dict) -> None:
		"""public method to set the scoreboard dict to a new dict"""
		self._scoreboard_dict = new_scoreboard_dict
