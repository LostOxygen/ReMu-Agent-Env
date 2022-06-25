import torch
from ..enums import EnumAction

from ..client.behavior import Behavior
from ..utils import override, render_to_surface, surface_to_tensor, convert_to_greyscale
from ..constants import (
	HEIGHT,
	WIDTH,
	LOG_EVERY,
	USE_REPLAY_AFTER
)

from .dqn_utils import (
	gamestate_to_tensor,
	gamestate_to_tensor_relative,
	get_nearest_neighbour,
	get_dist,
	get_angle,
	normalize_vals,
	plot_metrics,
	log_metrics
)
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

	def __init__(self, player_name: str, agent_name: str, device: str="cpu"):
		self.player_name = player_name
		self.agent_name = agent_name

		self.device = device

		self.steps_done = 0
		self.running_loss = 0

		self.optimizer = None

		self.last_score = 0
		self.last_max_q_value = 0.
		self.last_loss = 0.
		self.last_gamestate_tensor = None

		# metrics for plotting
		self.metrics_scores = []
		self.metrics_losses = []

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
			nearest_player = get_nearest_neighbour(self.player_name, players)
			# obtain the angle and distance towards the nearest player
			angle = get_angle(gamestate_tensor[0], nearest_player)
			dist = get_dist(gamestate_tensor[0], nearest_player)

			# normalize the values to be in the same range between [100, 0]
			max_dist = (torch.tensor([0, 0]) - torch.tensor([WIDTH, HEIGHT])).pow(2).sum().sqrt()
			angle, dist = normalize_vals(angle, 180, dist, max_dist)

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
		if loss is None:
			loss = self.last_loss
		if max_q_value is None:
			max_q_value = self.last_max_q_value

		self.steps_done += 1
		self.running_loss += loss
		print(f"loss: {(self.running_loss/self.steps_done):8.2f}\teps: {eps:8.2f} "
			  f"\tmax_q_value: {max_q_value:8.2f}\tsteps: {self.steps_done}", end="\r")

		# save the current state and actions for the next iteration
		self.last_loss = loss
		self.last_score = new_score
		self.last_gamestate_tensor = gamestate_tensor
		self.last_max_q_value = max_q_value

		# plot and log the metrics every LOG_EVERY steps
		if self.steps_done % LOG_EVERY == 0 and self.steps_done >= USE_REPLAY_AFTER:
			self.metrics_losses.append(loss)
			self.metrics_scores.append(scoreboard[self.player_name])
			log_metrics(new_score, loss, self.steps_done, self.player_name)
			plot_metrics(self.metrics_scores, self.metrics_losses, self.player_name)

		# return the action enum with the highest q value as a set
		return {predicted_action.to_enum_action()}
