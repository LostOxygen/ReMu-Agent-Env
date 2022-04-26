import json
from typing import Tuple

from ..enums import EnumAction

def deserialize_action(json_string: str) -> Tuple[str, list[EnumAction]]:
	'''
	Deserializes an action from a json string

	Parameters:
		json_string: a string in json format

	Returns:
		(name of the player, list of actions)
	'''

	obj = json.loads(json_string)

	player_name = obj["player_name"]
	actions = list(map(_string_to_action, obj["action"]))

	return (player_name, actions)

def _string_to_action(value: str) -> EnumAction:
	match value.lower():
		case "left":
			return EnumAction.LEFT
		case "forward":
			return EnumAction.FORWARD
		case "right":
			return EnumAction.BACKWARDS
		case "shoot":
			return EnumAction.SHOOT

	raise ValueError(f"Unknown action {value}")
