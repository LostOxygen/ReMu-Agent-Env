import logging
from copy import deepcopy
import torch

from ..enums import EnumAction

from .behavior import Behavior
from ..utils import override
from ..dqn_utils import MODEL_PATH, gamestate_to_tensor, get_model, save_model


class DqnBehavior(Behavior):
	"""DQN Behavior"""

	def __init__(self, player_name: str):
		self.player_name = player_name

		# model stuff
		self.model: torch.nn.Sequential = None
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.optimizer = Optimizer(self.player_name)

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
		gamestate_tensor = gamestate_to_tensor(self.player_name, players, projectiles, self.device)
		gamestate_tensor = torch.flatten(gamestate_tensor)

		# check if the model is already loaded, if not load it
		if self.model is None:
			self.model = get_model(self.device, len(gamestate_tensor), len(EnumAction), self.player_name)
			self.optimizer.init_networks(self.model)

		# let the network predict (outputs an tensor with q-values for all actions)
		prediction = self.model(gamestate_tensor) # pylint: disable=not-callable
		predicted_action = EnumAction(prediction.argmax(0).item())

		# run the next training step
		self.optimizer.apply_training_step(gamestate_tensor, reward, predicted_action)

		# save the current state and actions for the next iteration
		self.last_score = new_score

		# return the action enum with the highest q value as a set
		return {predicted_action}

class Optimizer:
	'''
	Applies optimization steps to a given network based on the deep q-learning training loop.
	'''

	def __init__(self,
		model_name: str,
		num_episodes=0,
		episodes_per_update=1000
	):
		'''
		Parameters:
			model_name: name of the model
			num_episodes: how many iterations should be performed or zero to run infinite
			episodes_per_update: after how many interations the target should be updated and saved
		'''

		self.model_name = model_name
		self.num_episodes = num_episodes
		self.episodes_per_update = episodes_per_update

		self.current_episode = 1
		self.last_state = None

	def init_networks(self, network: torch.nn.Module):
		'''
		Initializes the policy and target network. Must be called before using the
		"apply_training_step" method.

		Parameters:
			network: the network should be optimized
		'''

		self.target_network = deepcopy(network)
		self.policy_network = network

	def apply_training_step(self, state: torch.tensor, reward: int, action: EnumAction):
		'''
		Applies a training step to the model.

		Parameters:
			state: the new game state
			reward: the reward that was gained
			action: the next action that should be performed
		'''

		if self.target_network is None or self.policy_network is None:
			return

		if self.num_episodes != 0 and self.current_episode > self.num_episodes:
			# training is over, return
			return

		# todo: apply some optimization here

		self.last_state = state

		if self.current_episode % self.episodes_per_update == 0:
			self._update_target_network()

		self.current_episode += 1

	def _update_target_network(self):
		self.target_network.load_state_dict(self.policy_network.state_dict())
		save_model(self.target_network, MODEL_PATH+self.model_name)

		logging.info("Saved target network with name %s", self.model_name)
