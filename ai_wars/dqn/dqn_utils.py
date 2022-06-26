"""library for DQN utilities and classes"""
import os
import math
from typing import Tuple
from datetime import datetime
import logging
import torch
from torch import nn
from torchsummary import summary
from pygame.math import Vector2
import matplotlib.pyplot as plt

import ai_wars.constants
from ..constants import (
	MODEL_PATH,
	MAX_NUM_PROJECTILES,
	NUM_PLAYERS,
	DQN_PARAMETER_DICT,
    HIDDEN_NEURONS,
	MAX_ITERATIONS
)
from .dqn_models import DQNModelLinear, DQNModelLSTM, DQNModelCNN

UP = Vector2(0, -1)

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
	for i, projectile in enumerate(filter(lambda p: p["owner"] != own_name, projectiles)):
		if i < MAX_NUM_PROJECTILES:
			gamestate_tensor[NUM_PLAYERS + i, 0] = projectile["position"].x
			gamestate_tensor[NUM_PLAYERS + i, 1] = projectile["position"].y
			gamestate_tensor[NUM_PLAYERS + i, 2] = projectile["direction"].x
			gamestate_tensor[NUM_PLAYERS + i, 3] = projectile["direction"].y

	#return torch.round(gamestate_tensor, decimals=-1)
	return gamestate_tensor

def gamestate_to_tensor_relative(
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
		size=(NUM_PLAYERS-1 + MAX_NUM_PROJECTILES, 4),
		dtype=torch.float32,
		device=device
	)

	players_copy = players.copy()

	own_player = next(player for player in players_copy if player["player_name"] == own_name)
	position = own_player["position"]
	angle = own_player["direction"].angle_to(UP)

	# iterate over the remaining players
	for i, player in enumerate(filter(lambda p: p["player_name"] != own_name, players_copy)):
		relative_position = (player["position"] - position).rotate(angle)
		relative_direction = player["direction"].rotate(angle)

		gamestate_tensor[i, 0] = relative_position.x
		gamestate_tensor[i, 1] = relative_position.y
		gamestate_tensor[i, 2] = relative_direction.x
		gamestate_tensor[i, 3] = relative_direction.y

	for i, projectile in enumerate(filter(lambda p: p["owner"] != own_name, projectiles)):
		if i < MAX_NUM_PROJECTILES:
			relative_position = (projectile["position"] - position).rotate(angle)
			relative_direction = projectile["direction"].rotate(angle)

			gamestate_tensor[NUM_PLAYERS-1 + i, 0] = relative_position.x
			gamestate_tensor[NUM_PLAYERS-1 + i, 1] = relative_position.y
			gamestate_tensor[NUM_PLAYERS-1 + i, 2] = relative_direction.x
			gamestate_tensor[NUM_PLAYERS-1 + i, 3] = relative_direction.y

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


def get_nearest_neighbour(own_name: str, players: dict) -> Tuple[float, float, float, float]:
	"""
	Iterates over every enemy and returns the coords and rotation of the nearest neighbour

	Parameters:
		torch.tensor: Gamestate

	Returns:
		(float, float, float, float): (x, y, x_direction, y_direction)
	"""
	players_copy = players.copy()
	best_distance = float("inf")
	nearest_player = torch.tensor([0., 0.], dtype=torch.float)
	own_player = next(player for player in players_copy if player["player_name"] == own_name)
	own_player_vec = torch.tensor([own_player["position"].x,
								   own_player["position"].y],
								   dtype=torch.float)

	# iterate over every other "player" tensor in the whole gamestate
	for player in players_copy:
		if player["player_name"] != own_name:
			tmp_player_vec = torch.tensor([player["position"].x,
                                  		   player["position"].y], dtype=torch.float)

			tmp_distance = (own_player_vec - tmp_player_vec).pow(2).sum().sqrt()
			if tmp_distance < best_distance:
				nearest_player = torch.tensor([player["position"].x, player["position"].y])

	return nearest_player


def get_dist(own_player: torch.tensor, player: torch.tensor) -> float:
	"""
	Returns the distance between the own ship and the given player

	Parameters:
		torch.tensor([float, float, float, float]): own_player with(x, y, x_dir, y_dir)
		torch.tensor([float, float, float, float]): player with(x, y, x_dir, y_dir)

	Returns:
		float: distance
	"""
	own_player_vec = torch.tensor(
		[own_player[0], own_player[1]], dtype=torch.float)
	player_vec = torch.tensor([player[0], player[1]], dtype=torch.float)

	return (own_player_vec - player_vec).pow(2).sum().sqrt()


def get_angle(own_player: torch.tensor, player: torch.tensor):
	"""
	Returns the angle between the own ship and the given player

	Parameters:
		torch.tensor([float, float, float, float]): own_player with(x, y, x_dir, y_dir)
		torch.tensor([float, float, float, float]): player with(x, y, x_dir, y_dir)

	Returns:
		float: angle
	"""
	player_coords = torch.tensor([player[0], player[1]], dtype=torch.float)
	own_coords = torch.tensor([own_player[0], own_player[1]], dtype=torch.float)
	vector_between_player = player_coords-own_coords
	own_direction = torch.tensor([own_player[2], own_player[3]], dtype=torch.float)

	inner_product = torch.inner(vector_between_player, own_direction)
	player_norm = torch.linalg.vector_norm(vector_between_player)
	own_norm = torch.linalg.vector_norm(own_direction)

	# prevent zero division
	if player_norm == 0.0:
		player_norm += 1e-8
	if own_norm == 0.0:
		own_norm += 1e-8

	cos = inner_product / (player_norm * own_norm)
	angle = torch.acos(torch.clamp(cos, -1+1e-8, 1-1e-8))

	assert math.isnan(angle) is False, "Player angle is NaN"
	return angle


def normalize_vals(
	val1: float,
	limit1: float,
	val2: float,
	limit2: float
) -> Tuple[float, float]:
	"""
	Normalizes two values to a new range of [MAX, 0]

	Parameters:
		val1: first value
		val2: second value

	Returns:
		Tuple[float, float]: (val1, val2)
	"""
	new_val1 = ((100-0)*(val1-0) / limit1-0)+0  # distance
	new_val2 = ((100-0)*(val2-0) / limit2-0)+0  # angle

	return new_val1, new_val2


def plot_metrics(scores: torch.Tensor, losses: torch.Tensor, model_name: str) -> None:
	"""
	Plots score and loss for a model over epochs and saves the plot under ./plots/

	Parameters:
		scores: torch.Tensor with the current scores and their associated epochs of a given model
		losses: torch.Tensor with the current loss the associated epochs of a given model
		epoch: current epoch
		model_name: the name of the model

	Returns:
		None
	"""
	if not os.path.exists("plots/"):
		os.mkdir("plots/")

	# plot the scores
	plt.plot(scores)
	plt.title(f"{model_name} Score Metrics")
	plt.ylabel("Score")
	plt.xlabel("Epochs")
	plt.savefig(f"./plots/{model_name}_score.png")
	plt.close()

	# plot the losses
	plt.plot(losses)
	plt.title(f"{model_name} Loss Metrics")
	plt.ylabel("Score")
	plt.xlabel("Epochs")
	plt.savefig(f"./plots/{model_name}_loss.png")
	plt.close()


def log_metrics(score: torch.Tensor, loss: torch.Tensor, epoch: int, model_name: str) -> None:
	"""
	Logs score and loss for a model over epochs and saves the log under ./logs/model_name.log

	Parameters:
		scores: torch.Tensor with the current scoreof a given model
		loss: torch.Tensor with the current loss of a given model
		epoch: current epoch
		model_name: the name of the model

	Returns:
		None
	"""
	if epoch <= MAX_ITERATIONS:
		if not os.path.exists("logs/"):
			os.mkdir("logs/")

		try:
			with open(f"./logs/{model_name}.log", encoding="utf-8", mode="a") as log_file:
				log_file.write(f"{datetime.now().strftime('%A, %d. %B %Y %I:%M%p')} - epoch: {epoch} " \
							f"- score: {score} - loss: {loss:.2f}\n")
		except OSError as error:
			logging.error("Could not write logs into /logs/%s.log - error: %s", model_name, error)
