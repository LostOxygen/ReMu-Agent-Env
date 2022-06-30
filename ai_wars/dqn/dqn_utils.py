"""library for DQN utilities and classes"""
import os
import logging
import torch
from torch import nn
from torchsummary import summary
from pygame.math import Vector2

from ai_wars.maps.map import Map

import ai_wars.constants
from ..constants import (
	HEIGHT,
	MODEL_PATH,
	DQN_PARAMETER_DICT,
    HIDDEN_NEURONS,
	WIDTH
)
from .dqn_models import DQNModelLinear, DQNModelLSTM, DQNModelCNN

UP = Vector2(0, -1)

def gamestate_to_tensor(
	own_name: str,
	players: list[dict[str, any]],
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
		size=(1, 4),
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
	#return torch.round(gamestate_tensor, decimals=-1)
	return gamestate_tensor

def raycast_scan(
	origin: Vector2,
	game_map: Map,
	num_rays=8, step_size=1
) -> torch.tensor:
	def is_in_game_area(pos: Vector2) -> bool:
		return pos.x > 0 and pos.x < WIDTH and pos.y > 0 and pos.y < HEIGHT

	def cast_ray(angle: float):
		ray_pos = origin.copy()
		step = Vector2(0, -step_size).rotate(angle)

		while is_in_game_area(ray_pos) and game_map.is_point_in_bounds(ray_pos):
			ray_pos += step

		return origin.distance_to(ray_pos)

	angles = [360 / num_rays * i for i in range(num_rays)]
	values = list(map(cast_ray, angles))

	return torch.tensor(values, dtype=torch.float32)

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

	# logging.info("Saved target network with name %s", name)


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
	if ai_wars.constants.PARAM_SEARCH:
		hidden_neurons = DQN_PARAMETER_DICT[player_name]["hidden_neurons"]
	else:
		hidden_neurons = HIDDEN_NEURONS

	# create an empty new model
	model = DQNModelLinear(input_dim, hidden_neurons, output_dim)
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
