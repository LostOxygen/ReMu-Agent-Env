"""scoreboard library"""
from typing import List, Dict

from ai_wars.constants import START_SCORE
import pygame

class ScoreboardEntry:
	"""Objects stored inside the scoreboard."""

	score: int
	attempts: int
	finish_reached: int

	def __init__(self, score=0, attempts=0, finish_reached=0):
		self.score = score
		self.attempts = attempts
		self.finish_reached = finish_reached

	def __eq__(self, other):
		if not isinstance(other, ScoreboardEntry):
			return False

		return self.score == other.score and self.finish_reached == other.finish_reached

	def __str__(self):
		return f"finished {self.finish_reached} | score {self.score}"

class Scoreboard:
	"""The Scoreboard subject notifies observers when the state changes."""

	_observers: List
	_scoreboard_dict: Dict[str, ScoreboardEntry]

	def __init__(self):
		pygame.font.init()
		self.font = pygame.font.SysFont("consolas", 15)
		self.font_width = self.font.size("X")[0]
		self.font_height = self.font.size("X")[1]

		self._scoreboard_dict = {}

	def attach(self, observer) -> None:
		self._scoreboard_dict[observer.name] = ScoreboardEntry(START_SCORE, 0)

	def detach(self, observer) -> None:
		del self._scoreboard_dict[observer.name]

	def update_score(self, player_name: str, new_score: int) -> None:
		self._scoreboard_dict[player_name].score = new_score

	def increase_attempts(self, player_name: str):
		self._scoreboard_dict[player_name].attempts += 1

	def increment_finish_reached(self, player_name: str):
		self._scoreboard_dict[player_name].finish_reached += 1

	def draw_scoreboard(self, screen: pygame.Surface) -> None:
		"""public method to draw the scoreboard on the given screen"""
		for pos, (player, entry) in enumerate(self._scoreboard_dict.items()):
			score_string = f"{player}: attempts {entry.attempts}, finished {entry.finish_reached}, score {int(entry.score)}"
			text_surface = self.font.render(score_string, False, (0, 0, 0))
			screen.blit(text_surface, (0, self.font_height*pos))

	def get_scoreboard_dict(self) -> Dict[str, int]:
		"""public method to get the scoreboard dict"""
		return self._scoreboard_dict

	def set_scoreboard_dict(self, new_scoreboard_dict: dict) -> None:
		"""public method to set the scoreboard dict to a new dict"""
		self._scoreboard_dict = new_scoreboard_dict

	def _sort_entries(self):
		self._scoreboard_dict = dict(sorted(self._scoreboard_dict.items(),
			key=lambda x: x[1], reverse=True))
