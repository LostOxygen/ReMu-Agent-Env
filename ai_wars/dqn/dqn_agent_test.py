import abc
import torch

from ai_wars.utils import override


from . import AgentNotFoundException
from .dqn_utils import get_model_linear

from ..enums import MoveSet

from ..constants import (
	MOVEMENT_SET
)

def get_agent_test(agent_name: str, device: str, model_name: str, input_dim: int):
	'''
	Returns a agent for the given name.

	Throws AgentNotFoundException if no agent with given name exists.
	'''

	match agent_name:
		case "linear":
			print("here")
			return LinearAgentTest(device, model_name, input_dim)
		# case "lstm":
		# 	return LSTMAgent(device, model_name, input_dim)
		# case "cnn":
		# 	return CNNAgent(device, model_name, input_dim)

	raise AgentNotFoundException(agent_name)

class TestAgent(abc.ABC):

	def __init__(self,
		device: str,
		model_name: str,
		input_dim: int,
		load_model
	):
		'''
		Parameters:
			device: torch device to use
			model_name: name of the model
			num_episodes: how many iterations should be performed or zero to run infinite
			load_model: functions that provides the model
		'''

		self.device = device
		self.model_name = model_name

		self.network = load_model(device, input_dim, model_name)

	def _load_model(self, device, input_dim, model_name):
		return get_model_linear(device, input_dim, len(MOVEMENT_SET), model_name)

	@abc.abstractmethod
	def select_action(self, state: torch.tensor) -> MoveSet:
		'''
		Selects a actions based on the current game state.

		Parameters:
			state: current game state

		Returns:
			the selected action
		'''

		pass

class LinearAgentTest(TestAgent):

	def __init__(self,
		device: str,
		model_name: str,
		input_dim: int,
	):
		super().__init__(device, model_name, input_dim, self._load_model)

	@override
	def select_action(self, state):
		with torch.no_grad():
			pred = int(self.network(state).argmax())
		if pred not in range(len(MOVEMENT_SET)):
			return None

		return MOVEMENT_SET(pred)
