"""library for DQN utilities and classes"""
import os
import logging
import torch
from torch import nn
from torchsummary import summary

from ..constants import (
	BATCH_SIZE,
	MODEL_PATH,
	MAX_NUM_PROJECTILES,
	NUM_PLAYERS
)
from .dqn_models import DQNModelLinear, DQNModelLSTM, DQNModelCNN

def gamestate_to_tensor(
	own_name: str,
	players: list[dict[str, any]],
	projectiles: list[dict[str, any]],
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
		size=(NUM_PLAYERS + MAX_NUM_PROJECTILES, 4),
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
		if projectile["owner"] != own_name and i < MAX_NUM_PROJECTILES:
			gamestate_tensor[i + len(players), 0] = projectile["position"].x
			gamestate_tensor[i + len(players), 1] = projectile["position"].y
			gamestate_tensor[i + len(players), 2] = projectile["direction"].x
			gamestate_tensor[i + len(players), 3] = projectile["direction"].y

	#return torch.round(gamestate_tensor, decimals=-1)
	return gamestate_tensor


def save_model(model: nn.Sequential, name: str) -> None:
	"""
	Helper function to save a pytorch model to a specific path.

	Arguments:
		model: Pytorch Sequential Model
		path:  Path-string where the model should be saved

	Returns:
		None
	"""
	path = MODEL_PATH + name

	# check if the path already exists, if not create it
	if not os.path.exists(os.path.split(path)[0]):
		os.mkdir(os.path.split(path)[0])

	# extract the state_dict from the model to save it
	model_state = {
		"model": model.state_dict()
	}

	torch.save(model_state, path)

	logging.info("Saved target network with name %s", name)


def get_model_linear(device: str, input_dim: int, output_dim: int, player_name: str) -> nn.Module:
	"""
	Helper function to create a new model and copy it onto a specific device.

	Arguments:
		device: device string
		input_dim: input dimension of the model
		output_dim: output dimension of the model
		player_name: name of the network

	Returns:
		model: Pytorch Sequential Model
	"""
	loading_path = MODEL_PATH+player_name

	# create an empty new model
	model = DQNModelLinear(input_dim, output_dim)
	logging.debug("Created new model on %s", device)

	# check if a model with the player_name already exists and load it
	if os.path.isfile(loading_path):
		model_state = torch.load(loading_path, map_location=lambda storage, loc: storage)
		model.load_state_dict(model_state["model"], strict=True)
		logging.debug("Loaded model from %s", loading_path)

	logging.debug(summary(model, (input_dim,), device="cpu"))

	return model.to(device)


def get_model_cnn(device: str, input_dim: int, output_dim: int, player_name: str) -> nn.Module:
	"""
	Helper function to create a new model and copy it onto a specific device.

	Arguments:
		device: device string
		input_dim: input dimension of the model
		output_dim: output dimension of the model
		player_name: name of the network

	Returns:
		model: Pytorch Sequential Model
	"""
	loading_path = MODEL_PATH+player_name

	# create an empty new model
	model = DQNModelCNN(input_dim, output_dim)
	logging.debug("Created new model on %s", device)

	# check if a model with the player_name already exists and load it
	if os.path.isfile(loading_path):
		model_state = torch.load(
			loading_path, map_location=lambda storage, loc: storage)
		model.load_state_dict(model_state["model"], strict=True)
		logging.debug("Loaded model from %s", loading_path)

	logging.debug(summary(model, (1, 75, 100), device="cpu"))

	return model.to(device)


def get_model_lstm(device: str, num_features: int, sequence_length: int,
				   output_dim: int, player_name: str) -> nn.Module:
	"""
	Helper function to create a new model and copy it onto a specific device.

	Arguments:
		device: device string
		input_dim: input dimension of the model
		output_dim: output dimension of the model
		player_name: name of the network

	Returns:
		model: Pytorch Sequential Model
	"""
	loading_path = MODEL_PATH+player_name

	# create an empty new model
	model = DQNModelLSTM(num_features, sequence_length, output_dim)
	logging.debug("Created new model on %s", device)

	# check if a model with the player_name already exists and load it
	if os.path.isfile(loading_path):
		model_state = torch.load(loading_path, map_location=lambda storage, loc: storage)
		model.load_state_dict(model_state["model"], strict=True)
		logging.debug("Loaded model from %s", loading_path)

	return model.to(device)
