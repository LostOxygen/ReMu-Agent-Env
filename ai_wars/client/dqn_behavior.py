from queue import Queue
import torch

from ..enums import EnumAction

from .behavior import Behavior
from ..utils import override
from ..dqn_utils import gamestate_to_tensor, get_model


class DqnBehavior(Behavior):
	"""DQN Behavior"""

	def __init__(self, player_name: str, training_queue: Queue):
		self.player_name = player_name

		# model stuff
		self.model: torch.nn.Sequential = None
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# training and state stuff
		self.training_queue = training_queue
		self.last_score = 0
		self.last_state_tensor = None
		self.last_action = []

	@override
	def make_move(self,
		players: dict[str, any],
		projectiles: dict[str, any],
		scoreboard: dict[str, int]
	) -> set[EnumAction]:

		# obtain the new score and calculate the reward
		new_score = scoreboard[self.player_name]
		# reward = new_score - self.last_score

		# self.training_queue.put(self.last_state_tensor,
		# 						self.last_action, new_state_vector, reward)

		# prepare the gamestate for the model
		gamestate_tensor = gamestate_to_tensor(self.player_name, players, projectiles, self.device)
		gamestate_tensor = torch.flatten(gamestate_tensor)

		# check if the model is already loaded, if not load it
		if self.model is None:
			self.model = get_model(self.device, len(gamestate_tensor), len(EnumAction), self.player_name)

		# let the network predict (outputs an tensor with q-values for all actions)
		prediction = self.model(gamestate_tensor) # pylint: disable=not-callable
		predicted_action = EnumAction(prediction.argmax(0).item())

		# save the current state and actions for the next iteration
		self.last_score = new_score
		self.last_state_tensor = gamestate_tensor
		self.last_action = predicted_action

		# return the action enum with the highest q value as a set
		return {predicted_action}
