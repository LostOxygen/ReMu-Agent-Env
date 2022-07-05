"""library for DQN utilities and classes"""
import os
import logging
from copy import copy
from typing import Callable
import torch
from torch import nn
from torchsummary import summary
from pygame.math import Vector2

from ai_wars.maps.map import Map

from ai_wars.utils import convert_to_greyscale, render_to_surface, surface_to_tensor
from ai_wars import constants
from ..constants import (
	HEIGHT,
	MODEL_PATH,
	DQN_PARAMETER_DICT,
	HIDDEN_NEURONS,
	WIDTH,
	GAMESTATE_TO_INPUT
)
from .dqn_models import DQNModelLinear, DQNModelLSTM, DQNModelCNN

UP = Vector2(0, -1)

def gamestate_to_tensor(
	player_name: str,
	game_map: Map,
	players: dict[str, any],
	device="cpu"
) -> torch.tensor:
	match GAMESTATE_TO_INPUT:
		case "absolute_coordinates":
			player = list(filter(lambda p: p["player_name"] == player_name, players))[0]
			return absolute_coordinates(player, device)
		case "raycast_scan":
			player = list(filter(lambda p: p["player_name"] == player_name, players))[0]
			player_pos = player["position"]
			player_angle = player["direction"].angle_to(UP)
			return raycast_scan(player_pos, player_angle, game_map, device=device)
		case "cnn":
			return pygame_image(players, device)
	return None

def absolute_coordinates(
	player: dict[str, any],
	device: str = "cpu"
) -> torch.Tensor:
	"""
	Converts the gamestate to a  torch.Tensor. The tensor consists out of 4-tuples
	of the form (x, y, x_direction, y_direction) for every entity (like ships and bullets).

	Parameters:
		player: Information about the player
		device: The device the tensor should be stored on (cpu or cuda:0)

	Return:
		gamestate_tensor: the gamestate, converted to a torch.Tensor
	"""

	return torch.tensor([
		player["position"].x,
		player["position"].y,
		player["direction"].x,
		player["direction"].y
	], device=device)

def pygame_image(players: list[dict[str, any]], device="cpu"):
	"""
	Provides a screenshot of the current game state.

	Parameters:
		players: information about all players

	Returns:
		image as a tensor
	"""

	gamestate_surface = render_to_surface(players)
	gamestate_tensor = surface_to_tensor(gamestate_surface, device)
	return convert_to_greyscale(gamestate_tensor)

def raycast_scan(
	origin: Vector2,
	angle: float,
	game_map: Map,
	num_rays=8,
	step_size=1,
	draw_ray: Callable[[int, int], None] = lambda s, e: None,
	device="cpu"
) -> torch.tensor:
	"""
	Casts rays in call direction starting from a given point and rotation. Measures the distance to
	wall or border of the game field.

	Parameters:
		origin: Start point of the rays
		angle: Rotation of the origin
		game_map: Current game map
		num_rays: Number of rays that should be cast
		step_size: Smaller number is more accurate but slower to compute
		draw_ray: Callback for drawing the casted rays

	Returns:
		all measured distances as an 1d tensor
	"""

	def is_in_game_area(pos: Vector2) -> bool:
		return pos.x > 0 and pos.x < WIDTH and pos.y > 0 and pos.y < HEIGHT

	def cast_ray(angle: float):
		ray_pos = copy(origin)
		step = Vector2(0, -step_size).rotate(angle)

		while is_in_game_area(ray_pos) and game_map.is_point_in_bounds(ray_pos):
			ray_pos += step

		draw_ray(origin, ray_pos)

		return origin.distance_to(ray_pos)

	angles = [360 / num_rays * i - angle for i in range(num_rays)]
	values = list(map(cast_ray, angles))

	return torch.tensor(values, dtype=torch.float32, device=device)

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
	if constants.PARAM_SEARCH:
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
