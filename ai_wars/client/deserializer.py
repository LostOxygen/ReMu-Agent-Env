import json
from typing import Tuple

from pygame.math import Vector2

def deserialize_game_state(
	json_string: str
) -> Tuple[list[dict[str, any]], list[dict[str, any]], dict[str, int]]:
	'''
	Deserializes the current game state from a json string.

	Parameters:
		json_string: a string in json format

	Returns:
		(
			a list of players holding player_name (str), position (Vector2), direction (Vector2),
			a list of projectiles holding owner (str), position (Vector2), direction (Vector2),
			a dictionary of the scores of the players {name, score}
		)
	'''

	obj = json.loads(json_string)

	players = list(map(_dict_to_player, obj["players"]))
	projectiles = list(map(_dict_to_projectile, obj["projectiles"]))
	scoreboard = _dict_as_scoreboard(obj["scoreboard"])

	return (players, projectiles, scoreboard)

def _dict_to_vector(value: dict[str, float]) -> Vector2:
	x = value["x"]
	y = value["y"]

	return Vector2(x, y)

def _dict_to_player(value: dict[str, any]) -> dict[str, any]:
	name = value["name"]
	position = _dict_to_vector(value["position"])
	direction = _dict_to_vector(value["direction"])

	return {
		"player_name": name,
		"position": position,
		"direction": direction
	}

def _dict_to_projectile(value: dict[str, any]) -> dict[str, any]:
	owner = value["owner"]
	position = _dict_to_vector(value["position"])
	direction = _dict_to_vector(value["direction"])

	return {
		"owner": owner,
		"position": position,
		"direction": direction
	}

def _dict_as_scoreboard(value: list[dict[str, any]]) -> dict[str, int]:
	scoreboard = {}
	for entry in value:
		name = entry["name"]
		score = entry["score"]

		scoreboard[name] = score

	return scoreboard
