"""library for DQN utilities and classes"""
import os
import torch
from torch import nn

from .models import DQNModel

MAX_NUM_PROJECTILES = 128

def gamestate_to_tensor(
		own_name: str,
		players: list[dict[str, any]],
		projectiles: list[dict[str, any]],
    	scoreboard: dict[str, int], # pylint: disable=unused-argument
		device: str = "cpu"
	) -> torch.Tensor:
	"""
	Converts the gamestate to a  torch.Tensor. The tensor consists out of 4-tuples
	of the form (x, y, x_direction, y_direction) for every entity (like ships and bullets).

	Parameters:
		players: The players with their coordinates and directions
		projectiles: The projectiles with their coordinates and directions
		scoreboard: The scoreboard dictionary with the scores of the players
		device: The device the tensor should be stored on (cpu or cuda:0)

	Return:
		gamestate_tensor: the gamestate, converted to a torch.Tensor
	"""
	gamestate_tensor = torch.zeros(
		size=(len(players) + MAX_NUM_PROJECTILES, 4),
		dtype=torch.float32,
		device=device
	)

	# iterate over all players and save their position as well as their direction as 4-tuples
	# the own player is always at index 0
	players_copy = players.copy()
	for i, player in enumerate(players_copy):
		if player["player_name"] == own_name:
			gamestate_tensor[0, 0] = player["position"].x
			gamestate_tensor[0, 1] = player["position"].y
			gamestate_tensor[0, 2] = player["direction"].x
			gamestate_tensor[0, 3] = player["direction"].y
			# remove the own player from the list
			del players_copy[i]
			break

	# iterate over the remaining players
	for i, player in enumerate(players_copy):
		gamestate_tensor[i+1, 0] = player["position"].x
		gamestate_tensor[i+1, 1] = player["position"].y
		gamestate_tensor[i+1, 2] = player["direction"].x
		gamestate_tensor[i+1, 3] = player["direction"].y

	# iterate over all projectiles and save their position as well as their direction as 4-tuples
	# a maximum of MAX_NUM_PROJECTILES can be stored, while the rest is ignored
	# if there are less projectiles, the remaining indices stay filled with zeros
	for i, projectile in enumerate(projectiles):
		if i + len(players) < MAX_NUM_PROJECTILES:
			gamestate_tensor[i + len(players), 0] = projectile["position"].x
			gamestate_tensor[i + len(players), 1] = projectile["position"].y
			gamestate_tensor[i + len(players), 2] = projectile["direction"].x
			gamestate_tensor[i + len(players), 3] = projectile["direction"].y

	return gamestate_tensor


def save_model(model: nn.Sequential, path: str) -> None:
	"""
	Helper function to save a pytorch model to a specific path.

	Arguments:
		model: Pytorch Sequential Model
		path:  Path-string where the model should be saved

	Returns:
		None
	"""
	# check if the path already exists, if not create it
	if not os.path.exists(os.path.split(path)[0]):
		os.mkdir(os.path.split(path)[0])

	# extract the state_dict from the model to save it
	model_state = {
        "model": model.state_dict()
    }

	torch.save(model_state, path)


def load_model(model_path: str, device: str) -> nn.Sequential:
	"""
	Helper function to load a model state from a specific path into a model and copy it onto a device

	Arguments:
		model_path:  Path-string where the model should be loaded from
		device: device string

	Returns:
		model: Pytorch Sequential Model
	"""
	model = DQNModel()

	model_state = torch.load(model_path, map_location=lambda storage, loc: storage)
	model.load_state_dict(model_state["model"], strict=True)
	model = model.to(device)

	return model


def get_model(device: str) -> nn.Sequential:
	"""
	Helper function to create a new model and copy it onto a specific device.

	Arguments:
		device: device string

	Returns:
		model: Pytorch Sequential Model
	"""
	model = DQNModel()
	model = model.to(device)

	return model
