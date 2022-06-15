import abc
from collections import deque
import torch

from . import AgentNotFoundException
from .dqn_utils import get_model_linear, get_model_lstm

from ..utils import override
from ..enums import MoveSet

from ..constants import (
	MOVEMENT_SET,
	LSTM_SEQUENCE_SIZE
)

def get_agent_test(agent_name: str, device: str, model_name: str, input_dim: int):
	'''
	Returns a agent for the given name.

	Throws AgentNotFoundException if no agent with given name exists.
	'''

	match agent_name:
		case "linear":
			return LinearAgentTest(device, model_name, input_dim)
		case "lstm":
			return LstmAgentTest(device, model_name, input_dim)
		# case "cnn":
		# 	return CNNAgent(device, model_name, input_dim)

	raise AgentNotFoundException(agent_name)

class TestAgent(abc.ABC):
	'''
	Abstract DQN agent in testing mode
	'''

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
		input_dim: int
	):
		super().__init__(device, model_name, input_dim, self._load_model)

	@override
	def select_action(self, state):
		with torch.no_grad():
			pred = int(self.network(state).argmax())
		if pred not in range(len(MOVEMENT_SET)):
			return None

		return MOVEMENT_SET(pred)

class LstmAgentTest(TestAgent):

	def __init__(self,
		device: str,
		model_name: str,
		input_dim: int
	):
		super().__init__(device, model_name, input_dim, self._load_model)

		self.sequence_queue = deque(maxlen=LSTM_SEQUENCE_SIZE)

	def _load_model(self, device, input_dim, model_name):
		return get_model_lstm(device, input_dim, LSTM_SEQUENCE_SIZE, len(MOVEMENT_SET), model_name)

	@override
	def select_action(self, state):
		self.sequence_queue.append(state)

		if len(self.sequence_queue) >= LSTM_SEQUENCE_SIZE:
			with torch.no_grad():
				state_vector = torch.stack(list(self.sequence_queue))
				pred = int(self.network(state_vector).argmax())
		else:
			pred = None

		if pred not in range(len(MOVEMENT_SET)):
			return None
		return MOVEMENT_SET(pred)
