import json

from pygame.math import Vector2

from ai_wars.scoreboard import ScoreboardEntry

from ..spaceship import Spaceship

def serialize_game_state(
	spaceships: list[Spaceship],
	scoreboard: dict[str, ScoreboardEntry]
) -> str:
	'''
	Serializes the current game state into a json string

	Parameters:
		spaceships: list of spaceships
		bullets: list of bullets
		scoreboard: current scoreboard of the game

	Returns:
		json string
	'''

	player = list(map(_spaceship_as_dict, spaceships))
	scores = _scoreboard_as_dict(scoreboard)

	game_state = {
		"players": player,
		"scoreboard": scores
	}

	return json.dumps(game_state)

def _vector_as_dict(vec: Vector2):
	return {
		"x": vec.x,
		"y": vec.y
	}

def _spaceship_as_dict(spaceship: Spaceship) -> dict[str, any]:
	return {
		"name": spaceship.name,
		"position": _vector_as_dict(Vector2(spaceship.x, spaceship.y)),
		"direction": _vector_as_dict(spaceship.direction)
	}

def _scoreboard_as_dict(scoreboard: dict[str, ScoreboardEntry]) -> list[dict[str, any]]:
	def _entry_as_dict(name, entry):
		return {
			"name": name,
			"score": entry.score,
			"attempts": entry.attempts,
			"finish_reached": entry.finish_reached
		}

	return list(map(
		lambda e: _entry_as_dict(*e),
		scoreboard.items()
	))
