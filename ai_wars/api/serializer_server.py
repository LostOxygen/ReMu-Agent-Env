import json

from pygame.math import Vector2

from ..spaceship import Spaceship
from ..bullet import Bullet

def serialize_game_state(
	spaceships: list[Spaceship],
	bullets: list[Bullet],
	scoreboard: dict[str, int]
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
	bullets = list(map(_bullet_as_dict, bullets))
	scores = _scoreboard_as_dict(scoreboard)

	game_state = {
		"players": player,
		"projectiles": bullets,
		"scores": scores
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
		"direction": "todo"
	}

def _bullet_as_dict(bullet: Bullet) -> dict[str, any]:
	return {
		"owner": bullet.shooter.name,
		"position": _vector_as_dict(Vector2(bullet.x, bullet.y)),
		"direction": _vector_as_dict(bullet.velocity)
	}

def _scoreboard_as_dict(scoreboard: dict[str, int]) -> list[dict[str, any]]:
	def _entry_as_dict(name, score):
		return {
			"name": name,
			"score": score
		}

	return list(map(
		lambda e: _entry_as_dict(*e),
		scoreboard.items()
	))
