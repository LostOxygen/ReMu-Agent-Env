from ..enums import EnumAction

from ..client.behavior import Behavior
from ..utils import override

from .dqn_utils import gamestate_to_tensor
from .dqn_agent import get_agent

class DqnBehavior(Behavior):
	"""DQN Behavior"""

	def __init__(self, player_name: str, agent_name: str, device="cpu"):
		self.player_name = player_name
		self.agent_name = agent_name
		self.device = device

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
		gamestate_tensor = gamestate_to_tensor(self.player_name, players, projectiles, self.device) \
			.flatten()

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
		print(f"loss: {loss:8.2f}\teps: {eps:8.2f}\tmax q value: {max_q_value:8.2f}", end="\r")

		# save the current state and actions for the next iteration
		self.last_score = new_score

		# return the action enum with the highest q value as a set
		return {predicted_action}
