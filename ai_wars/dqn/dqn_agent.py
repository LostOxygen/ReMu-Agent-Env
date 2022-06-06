import abc
import random
from typing import Tuple
from collections import deque
from copy import deepcopy
import torch
from torch.nn import functional as F

from ..enums import MoveSet

from ..utils import override

from .dqn_utils import get_model_linear, get_model_lstm, get_model_cnn, save_model
from .replay_memory import ReplayMemory, Transition

from ..constants import (
	MOVEMENT_SET,
	MEMORY_SIZE,
	BATCH_SIZE,
	GAMMA,
	EPS_START,
	EPS_END,
	DECAY_FACTOR,
	LSTM_SEQUENCE_SIZE,
	UPDATE_EVERY,
	LEARNING_RATE,
	TAU,
	USE_REPLAY_AFTER
)


class AgentNotFoundException(Exception):

	def __init__(self, name):
		super().__init__(f"Model with name {name} not found!")

def get_agent(agent_name: str, device: str, model_name: str, input_dim: int):
	'''
	Returns a agent for the given name.

	Throws AgentNotFoundException if no agent with given name exists.
	'''

	match agent_name:
		case "linear":
			return LinearAgent(device, model_name, input_dim)
		case "lstm":
			return LSTMAgent(device, model_name, input_dim)
		case "cnn":
			return CNNAgent(device, model_name, input_dim)

	raise AgentNotFoundException(agent_name)

class Agent(abc.ABC):
	'''
	Abstract DQN agent.
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

		self.policy_network = load_model(device, input_dim, model_name)
		self.target_network = deepcopy(self.policy_network)

		self.memory = ReplayMemory(MEMORY_SIZE, BATCH_SIZE, self.device)
		self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=LEARNING_RATE)

		self.current_episode = 0
		self.eps = EPS_START
		self.t_step = 0

	def apply_training_step(self,
		state: torch.tensor,
		reward: int,
		action: MoveSet,
		next_state: torch.tensor
	) -> Tuple[float, float, float]:
		'''
		Applies a training step to the model.

		Parameters:
			state: the new game state
			reward: the reward that was gained
			action: the next action that should be performed

		Returns:
			(current loss, current epsilon, current best q value)
		'''

		if self.num_episodes != 0 and self.current_episode > self.num_episodes:
			# training is over, return
			return

		self.memory.add(state, action.value, reward, next_state)

		self.t_step = (self.t_step + 1) % UPDATE_EVERY

		if len(self.memory) >= USE_REPLAY_AFTER and self.t_step == 0:
			states, actions, rewards, next_states = self.memory.sample()

			state_action_values = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

			next_state_values = self.target_network(next_states).max(1)[0].detach()
			expected_state_action_values = (next_state_values * GAMMA) + rewards
			max_q_value = expected_state_action_values.max()

			loss = F.mse_loss(state_action_values, expected_state_action_values)

			# Optimize the model
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		else:
			loss = 0.
			max_q_value = 0.

		self._update_target_network()
		self.current_episode += 1
		return (loss, self.eps, max_q_value)


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

	@abc.abstractmethod
	def update_replay_memory(self, state: torch.tensor, reward: int, action: MoveSet):
		'''
		Replay memory should be updated here.

		Parameters:
			state: current game state
			reward: the last received reward
			action: the choosen action
		'''

		pass

	@abc.abstractmethod
	def post_step(self, state: torch.tensor):
		'''
		Operations after the training step may be applied here.

		Parameters:
			state: current game state
		'''

		pass

	def _update_target_network(self):
		# update the target networks paramters
		for target_param, local_param in zip(self.target_network.parameters(),
											 self.policy_network.parameters()):
			target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
		# save the taget network
		save_model(self.target_network, self.model_name)


class LinearAgent(Agent):
	'''
	Implementation of a DQN agent that uses a linear network.
	'''

	def __init__(self,
		device: str,
		model_name: str,
		input_dim: int,
		num_episodes=0,
		episodes_per_update=1000
	):
		super().__init__(device, model_name, input_dim, self._load_model)

		self.num_episodes = num_episodes
		self.episodes_per_update = episodes_per_update

		self.last_state = None

	def _load_model(self, device, input_dim, model_name):
		return get_model_linear(device, input_dim, len(MOVEMENT_SET), model_name)

	@override
	def select_action(self, state):
		sample = random.random()
		self.policy_network.eval()

		if sample > self.eps:
			with torch.no_grad():
				pred = int(self.policy_network(state).argmax())
		else:
			pred = random.randrange(len(MOVEMENT_SET))
		self.policy_network.train()

		if pred not in range(len(MOVEMENT_SET)):
			return None

		self.eps = max(EPS_END, DECAY_FACTOR * self.eps)
		return MOVEMENT_SET(pred)

	@override
	def update_replay_memory(self, state, reward, action):
		if self.last_state is None:
			self.last_state = torch.zeros(state.shape).to(self.device)
		self.memory.add(self.last_state, action.value, reward, state)

	@override
	def post_step(self, state):
		self.last_state = state


class LSTMAgent(Agent):
	'''
	Implementation of a DQN agent that uses a LSTM network.
	'''

	def __init__(self,
		device: str,
		model_name: str,
		input_dim: int,
		num_episodes=0,
		episodes_per_update=1000
	):
		super().__init__(device, model_name, input_dim, self._load_model)

		self.num_episodes = num_episodes
		self.episodes_per_update = episodes_per_update

		self.sequence_queue = deque(maxlen=LSTM_SEQUENCE_SIZE)
		self.last_sequence = torch.zeros((LSTM_SEQUENCE_SIZE, input_dim)).to(self.device)

	def _load_model(self, device, input_dim, model_name):
		return get_model_lstm(device, input_dim, LSTM_SEQUENCE_SIZE, len(MOVEMENT_SET), model_name)

	@override
	def select_action(self, state):
		sample = random.random()
		self.policy_network.eval()

		if len(self.sequence_queue) >= LSTM_SEQUENCE_SIZE and sample > self.eps:
			with torch.no_grad():
				prev_states = list(self.sequence_queue)[1:]
				prev_states.append(state)
				state_vector = torch.stack(prev_states)
				pred = int(self.policy_network(state_vector).argmax())
		else:
			pred = random.randrange(len(MOVEMENT_SET))
		self.policy_network.train()

		if pred not in range(len(MOVEMENT_SET)):
			return None
		return MOVEMENT_SET(pred)

	@override
	def update_replay_memory(self, state, reward, action): # pylint: disable=unused-argument
		if len(self.sequence_queue) >= LSTM_SEQUENCE_SIZE:
			sequence = torch.stack(list(self.sequence_queue))
			self.memory.push(Transition(self.last_sequence, action.value, sequence, reward))
			self.last_sequence = sequence

	@override
	def post_step(self, state):
		self.sequence_queue.append(state)


class CNNAgent(Agent):
	'''
	Implementation of a DQN agent that uses a cnn network.
	'''

	def __init__(self,
			  device: str,
			  model_name: str,
			  input_dim: int,
			  num_episodes=0,
			  episodes_per_update=1000
			  ):
		super().__init__(device, model_name, input_dim, self._load_model)

		self.num_episodes = num_episodes
		self.episodes_per_update = episodes_per_update

		self.last_state = None

	def _load_model(self, device, input_dim, model_name):
		return get_model_cnn(device, input_dim, len(MOVEMENT_SET), model_name)

	@override
	def select_action(self, state):
		sample = random.random()
		self.policy_network.train()

		if sample > self.eps:
			with torch.no_grad():
				pred = int(self.policy_network(state.unsqueeze(0)).argmax())
		else:
			pred = random.randrange(len(MOVEMENT_SET))
		self.policy_network.train()

		if pred not in range(len(MOVEMENT_SET)):
			return None
		return MOVEMENT_SET(pred)

	@override
	def update_replay_memory(self, state, reward, action):
		if self.last_state is None:
			self.last_state = torch.zeros(state.shape).to(self.device)
		self.memory.push(Transition(self.last_state, action.value, state, reward))

	@override
	def post_step(self, state):
		self.last_state = state
