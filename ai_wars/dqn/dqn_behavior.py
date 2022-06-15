from typing import Tuple
import math
import torch
from ..enums import EnumAction

from ..client.behavior import Behavior
from ..utils import override, render_to_surface, surface_to_tensor, convert_to_greyscale
from ..constants import (
	NUM_PLAYERS,
	HEIGHT,
	WIDTH
)

from .dqn_utils import gamestate_to_tensor, gamestate_to_tensor_relative
from .dqn_agent_test import get_agent_test
from .dqn_agent_train import get_agent

from ..constants import (
	RELATIVE_COORDINATES_MODE
)

class DqnBehaviorTest(Behavior):

	def __init__(self, player_name: str, agent_name: str, device="cpu"):
		self.player_name = player_name
		self.agent_name = agent_name
		self.device = device

		self.agent = None

	@override
	def make_move(self,
		players: dict[str, any],
		projectiles: dict[str, any],
		scoreboard: dict[str, int]
	) -> set[EnumAction]:
		if RELATIVE_COORDINATES_MODE:
			gamestate_tensor = gamestate_to_tensor_relative(self.player_name, players,
															projectiles, self.device)
		else:
			gamestate_tensor = gamestate_to_tensor(self.player_name, players, projectiles, self.device)
		gamestate_tensor = gamestate_tensor.flatten()

		if self.agent is None:
			self.agent = get_agent_test(self.agent_name, self.device,
								self.player_name, len(gamestate_tensor))

		predicted_action = self.agent.select_action(gamestate_tensor)
		if predicted_action is None:
			return {}
		return {predicted_action.to_enum_action()}

class DqnBehavior(Behavior):
	"""DQN Behavior"""

	def __init__(self, player_name: str, agent_name: str, device="cpu"):
		self.player_name = player_name
		self.agent_name = agent_name

		self.device = device

		self.steps_done = 0
		self.running_loss = 0

		self.optimizer = None

		self.last_score = 0
		self.last_gamestate_tensor = None

	@override
	def make_move(self,
		players: dict[str, any],
		projectiles: dict[str, any],
		scoreboard: dict[str, int]
	) -> set[EnumAction]:
		# prepare the gamestate for the model
		if self.agent_name == "cnn":
			gamestate_surface = render_to_surface(players, projectiles)
			gamestate_tensor = surface_to_tensor(gamestate_surface, self.device)
			gamestate_tensor = convert_to_greyscale(gamestate_tensor)
			# obtain the new score and calculate the reward
			new_score = scoreboard[self.player_name]
			reward = (new_score - self.last_score)
		else:
			if RELATIVE_COORDINATES_MODE:
				gamestate_tensor = gamestate_to_tensor_relative(self.player_name, players,
																projectiles, self.device)
			else:
				gamestate_tensor = gamestate_to_tensor(self.player_name, players, projectiles, self.device)
			# extract nearest player from gamestate
			nearest_player = self.get_nearest_neighbour(gamestate_tensor)
			# obtain the angle and distance towards the nearest player
			angle = self.get_angle(gamestate_tensor[0], nearest_player)
			dist = self.get_dist(gamestate_tensor[0], nearest_player)

			# normalize the values to be in the same range between [100, 0]
			max_dist = (torch.tensor([0, 0]) - torch.tensor([WIDTH, HEIGHT])).pow(2).sum().sqrt()
			angle, dist = self.normalize_vals(angle, 180, dist, max_dist)

			gamestate_tensor = gamestate_tensor.flatten()

			# obtain the new score and calculate the reward and subtract the distance and the angle
			new_score = scoreboard[self.player_name]
			reward = (new_score - self.last_score) + (100 - angle) + (100 - dist)
			reward = reward.item()

		# check if the model is already loaded, if not load it
		if self.optimizer is None:
			self.optimizer = get_agent(self.agent_name, self.device,
								self.player_name, len(gamestate_tensor))

		# let the network predict (outputs an tensor with q-values for all actions)
		predicted_action = self.optimizer.select_action(gamestate_tensor)
		if predicted_action is None:
			return {}

		if self.last_gamestate_tensor is None:
			self.last_gamestate_tensor = gamestate_tensor

		# run the next training step
		loss, eps, max_q_value = self.optimizer.apply_training_step(self.last_gamestate_tensor,
															 		reward, predicted_action,
                                                              		gamestate_tensor)
		self.steps_done += 1
		self.running_loss += loss
		print(f"loss: {(self.running_loss/self.steps_done):8.2f}\teps: {eps:8.2f} "
					f"\tmax_q_value: {max_q_value:8.2f}\tsteps: {self.steps_done}", end="\r")

		# save the current state and actions for the next iteration
		self.last_score = new_score
		self.last_gamestate_tensor = gamestate_tensor

		# return the action enum with the highest q value as a set
		return {predicted_action.to_enum_action()}

	def get_nearest_neighbour(
		self, gamestate_tensor: torch.tensor
	) -> Tuple[float, float, float, float]:
		"""
		Iterates over every enemy and returns the coords and rotation of the nearest neighbour

		Parameters:
			torch.tensor: Gamestate

		Returns:
			(float, float, float, float): (x, y, x_direction, y_direction)
		"""
		best_distance = float("inf")
		nearest_player = torch.tensor([0., 0., 0., 0.], dtype=torch.float)
		own_player_vec = torch.tensor(
			[gamestate_tensor[0][0], gamestate_tensor[0][1]], dtype=torch.float)
		# iterate over every other "player" tensor in the whole gamestate
		for player in gamestate_tensor[1:NUM_PLAYERS]:
			tmp_player_vec = torch.tensor([player[0], player[1]], dtype=torch.float)

			tmp_distance = (own_player_vec - tmp_player_vec).pow(2).sum().sqrt()
			if tmp_distance < best_distance:
				nearest_player = player

		return nearest_player

	def get_dist(self, own_player: torch.tensor, player: torch.tensor) -> float:
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

	def get_angle(self, own_player: torch.tensor, player: torch.tensor):
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

	def normalize_vals(self,
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
