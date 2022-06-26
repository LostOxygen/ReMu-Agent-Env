import abc
import random
import torch

from ai_wars.dqn.dqn_utils import get_oriented_angle, get_dist

from . import AgentNotFoundException

from ..utils import override
from ..enums import MoveSet

from ..constants import (
	MOVEMENT_SET,
	CHANGE_TARGET_PROB,
	ANGLE_TRESH_RIGHT,
    ANGLE_TRESH_LEFT,
	DIST_THRESH
)


def get_agent(agent_name: str, device: str, model_name: str): # pylint: disable=unused-argument
	'''
	Returns a agent for the given name.

	Throws AgentNotFoundException if no agent with given name exists.
	'''

	match agent_name:
		case "full_heuristic":
			return FullHeuristicAgent(device, model_name)
		case "heuristic":
			return FullHeuristicAgent(device, model_name)

	raise AgentNotFoundException(agent_name)


class Agent(abc.ABC):
	'''
	Abstract DQN agent in testing mode
	'''

	def __init__(self,
			  device: str,
			  model_name: str):
		'''
		Parameters:
			device: torch device to use
			model_name: name of the model
			num_episodes: how many iterations should be performed or zero to run infinite
			load_model: functions that provides the model
		'''

		self.device = device
		self.model_name = model_name
		self.current_target = None
		self.current_target_vec = None

	@abc.abstractmethod
	def select_action(self, players: dict[str, any], projectiles: dict[str, any]) -> MoveSet:
		'''
		Selects a actions based on the current game state.

		Parameters:
			players: dictionary with player names and their coordinates and rotation
			projectiles: dictionary with all projectile coordinates and their directions

		Returns:
			the selected action
		'''
		pass


class FullHeuristicAgent(Agent):
	"""Class for a fully heuristic agent"""

	def __init__(self, device: str, model_name: str): # pylint: disable=useless-super-delegation
		super().__init__(device, model_name)

	@override
	def select_action(self, players: list[dict[str, any]], projectiles: list[dict[str, any]]):
		own_player = next(player for player in players if player["player_name"] == self.model_name)
		own_player_vec = torch.tensor([own_player["position"].x,
									   own_player["position"].y,
									   own_player["direction"].x,
									   own_player["direction"].y],
									   dtype=torch.float)

		# remove own player from the player list
		filtered_players = [player for player in players if player["player_name"] != self.model_name]

		# check if there are enough players ingame
		if len(filtered_players) > 0:

			if self.current_target is None or random.random() > CHANGE_TARGET_PROB:
				# choose a random new target and create the target vector
				self.current_target = filtered_players[random.randint(0, len(filtered_players)-1)]
				self.current_target_vec = torch.tensor([self.current_target["position"].x,
														self.current_target["position"].y,
														self.current_target["direction"].x,
														self.current_target["direction"].y],
														dtype=torch.float)


			angle_to_target = get_oriented_angle(own_player_vec, self.current_target_vec)

			dist_to_target = get_dist(own_player_vec, self.current_target_vec)

			# turn right
			if angle_to_target > ANGLE_TRESH_RIGHT:
				return MOVEMENT_SET(1)

			# turn left
			elif angle_to_target < ANGLE_TRESH_LEFT:
				return MOVEMENT_SET(0)

			# move forward
			elif dist_to_target > DIST_THRESH:
				return MOVEMENT_SET(2)

			# shoot
			else:
				return MOVEMENT_SET(4)

		# if not, return a random action
		else:
			return MOVEMENT_SET(random.randint(0, len(MOVEMENT_SET)-1))
