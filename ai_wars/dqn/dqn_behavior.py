from pygame import Vector2

from ..enums import EnumAction

from ..maps.map import Map
from ..client.behavior import Behavior
from ..utils import override

from .dqn_utils import gamestate_to_tensor
from .dqn_agent_test import get_agent_test
from .dqn_agent_train import get_agent

from ..constants import MAX_ITERATIONS


UP = Vector2(0, -1)

class DqnBehaviorTest(Behavior):

	def __init__(self, player_name: str, agent_name: str, game_map: Map, device="cpu"):
		self.player_name = player_name
		self.agent_name = agent_name
		self.game_map = game_map
		self.device = device

		self.agent = None

	@override
	def make_move(self,
		players: dict[str, any],
		scoreboard: dict[str, int]
	) -> set[EnumAction]:
		gamestate_tensor = gamestate_to_tensor(self.player_name, self.game_map, players, self.device)

		if self.agent is None:
			self.agent = get_agent_test(self.agent_name, self.device,
								self.player_name, len(gamestate_tensor))

		predicted_action = self.agent.select_action(gamestate_tensor)
		if predicted_action is None:
			return {}
		return predicted_action.to_action_set()

class DqnBehavior(Behavior):
	"""DQN Behavior"""

	def __init__(self, player_name: str, agent_name: str, game_map: Map, device="cpu"):
		self.player_name = player_name
		self.agent_name = agent_name
		self.game_map = game_map
		self.device = device

		self.steps_done = 0
		self.running_loss = 0

		self.optimizer = None

		self.last_gamestate_tensor = None
		self.last_score = 0

	@override
	def make_move(self,
		players: dict[str, any],
		scoreboard: dict[str, int]
	) -> set[EnumAction]:
		if self.steps_done >= MAX_ITERATIONS:
			return {}

		gamestate_tensor = gamestate_to_tensor(self.player_name, self.game_map, players, self.device)

		# obtain the new score and calculate the reward and subtract the distance and the angle
		score = scoreboard[self.player_name].score
		if score > self.last_score:
			reward = 1000
		elif score < self.last_score:
			reward = -1000
		else:
			reward = 0

		min_val = gamestate_tensor.min()
		if min_val > 0:
			reward += int(min_val)

		# check if the model is already loaded, if not load it
		if self.optimizer is None:
			self.optimizer = get_agent(self.agent_name, self.device,
								self.player_name, len(gamestate_tensor))

		# let the network predict (outputs an tensor with q-values for all actions)
		predicted_action = self.optimizer.select_action(gamestate_tensor)
		if predicted_action is None:
			return {}

		# run the next training step
		loss, eps, max_q_value = \
			self.optimizer.apply_training_step(gamestate_tensor, reward, predicted_action)
		self.steps_done += 1
		self.running_loss += loss
		print(f"loss: {(self.running_loss/self.steps_done):8.2f}\teps: {eps:8.2f} "
					f"\tmax_q_value: {max_q_value:8.2f}\tsteps: {self.steps_done}", end="\r")

		# save the current state and actions for the next iteration
		self.last_gamestate_tensor = gamestate_tensor
		self.last_score = score

		# return the action enum with the highest q value as a set
		return predicted_action.to_action_set()
