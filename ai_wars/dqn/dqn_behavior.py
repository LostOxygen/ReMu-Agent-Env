import pygame
from ..enums import EnumAction

from ..client.behavior import Behavior
from ..utils import override, render_to_surface, surface_to_tensor, convert_to_greyscale

from .dqn_utils import gamestate_to_tensor
from .dqn_agent import get_agent

class DqnBehavior(Behavior):
	"""DQN Behavior"""

	def __init__(self, player_name: str, agent_name: str, device="cpu"):
		self.player_name = player_name
		self.agent_name = agent_name

		if self.agent_name == "cnn":
			pygame.init()
			pygame.display.init()
			pygame.display.set_mode((1, 1))

		self.device = device

		self.steps_done = 0
		self.running_loss = 0

		self.optimizer = None

		self.last_score = 0

	@override
	def make_move(self,
		players: dict[str, any],
		projectiles: dict[str, any],
		scoreboard: dict[str, int]
	) -> set[EnumAction]:
		# obtain the new score and calculate the reward
		new_score = scoreboard[self.player_name]
		reward = new_score - self.last_score

		# prepare the gamestate for the model
		if self.agent_name == "cnn":
			gamestate_surface = render_to_surface(players, projectiles)
			gamestate_tensor = surface_to_tensor(gamestate_surface, self.device)
			gamestate_tensor = convert_to_greyscale(gamestate_tensor)
		else:
			gamestate_tensor = gamestate_to_tensor(self.player_name, players, projectiles, self.device)
			gamestate_tensor = gamestate_tensor.flatten()

		# check if the model is already loaded, if not load it
		if self.optimizer is None:
			self.optimizer = get_agent(self.agent_name, self.device,
									   self.player_name, len(gamestate_tensor))

		# let the network predict (outputs an tensor with q-values for all actions)
		predicted_action = self.optimizer.select_action(gamestate_tensor)
		if predicted_action is None:
			return {}

		# run the next training step
		loss, eps, max_q_value = self.optimizer.apply_training_step(gamestate_tensor, reward,
																	predicted_action)
		self.steps_done += 1
		self.running_loss += loss
		print(f"loss: {(self.running_loss/self.steps_done):8.2f}\teps: {eps:8.2f} "\
			  f"\tmax q value: {max_q_value:8.2f}\tsteps: {self.steps_done}", end="\r")

		# save the current state and actions for the next iteration
		self.last_score = new_score

		# return the action enum with the highest q value as a set
		return {predicted_action}
