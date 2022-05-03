import json

from ..enums import EnumAction

def serialize_action(
	player_name: str,
	enum_actions: list[EnumAction]
) -> str:
	'''
	Serializes a list of actions a player performs into a json string.

	Parameters:
		player_name: the name of the player
		enum_actions: a list of actions

	Returns:
		json string
	'''
	actions = _enum_action_as_dict(enum_actions)

	action = {
		"name": player_name,
		"actions": actions
	}

	return json.dumps(action)

def _enum_action_as_dict(action: EnumAction) -> EnumAction:
	def _action_as_dict(action: EnumAction):
		match action:
			case EnumAction.LEFT: return "left"
			case EnumAction.FORWARD: return "forward"
			case EnumAction.RIGHT: return "right"
			case EnumAction.BACKWARD: return "backward"
			case EnumAction.SHOOT: return "shoot"

	return list(map(_action_as_dict, action))
