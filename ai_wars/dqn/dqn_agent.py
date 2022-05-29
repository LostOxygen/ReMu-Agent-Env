import abc
import random
from collections import deque
from copy import deepcopy
import torch

from ..utils import override

from ..enums import EnumAction

from .dqn_utils import get_model_linear, get_model_lstm, get_model_cnn, save_model
from .replay_memory import ReplayMemory, Transition

from ..constants import (
    MEMORY_SIZE,
    BATCH_SIZE,
    GAMMA,
    EPS_START,
    EPS_END,
    DECAY_FACTOR,
   	LSTM_SEQUENCE_SIZE
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

		self.memory = ReplayMemory(MEMORY_SIZE)
		self.optimizer = torch.optim.RMSprop(self.policy_network.parameters())

		self.current_episode = 1
		self.eps = EPS_START


	def apply_training_step(self, state: torch.tensor, reward: int, action: EnumAction):
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

		self.update_replay_memory(state, reward, action)

		if len(self.memory) >= BATCH_SIZE:
			transitions = self.memory.sample(BATCH_SIZE)
			batch = Transition(*zip(*transitions))

			state_batch = torch.stack(batch.state)
			action_batch = torch.tensor(batch.action).to(self.device).unsqueeze(0)
			reward_batch = torch.tensor(batch.reward).to(self.device).unsqueeze(0)

			state_action_values = self.policy_network(state_batch).gather(1, action_batch)

			next_state_values = self.target_network(state_batch).max(1)[0].detach()
			expected_state_action_values = (next_state_values * GAMMA) + reward_batch

			max_q_value = expected_state_action_values.max()

			criterion = torch.nn.SmoothL1Loss()
			loss = criterion(state_action_values, expected_state_action_values)

			# Optimize the model
			self.optimizer.zero_grad()
			loss.backward()
			for param in self.policy_network.parameters():
				param.grad.data.clamp_(-1, 1)
			self.optimizer.step()

			self.eps = max(EPS_END, DECAY_FACTOR * self.eps)
		else:
			loss = 0
			max_q_value = 0

		if self.current_episode % self.episodes_per_update == 0:
			self._update_target_network()

		self.post_step(state)

		self.current_episode += 1

		return loss, self.eps, max_q_value

	@abc.abstractmethod
	def select_action(self, state: torch.tensor) -> EnumAction:
		'''
		Selects a actions based on the current game state.

		Parameters:
			state: current game state

		Returns:
			the selected action
		'''

		pass

	@abc.abstractmethod
	def update_replay_memory(self, state: torch.tensor, reward: int, action: EnumAction):
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
		self.target_network.load_state_dict(self.policy_network.state_dict())
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
		return get_model_linear(device, input_dim, len(EnumAction), model_name)

	@override
	def select_action(self, state):
		sample = random.random()

		if sample > self.eps:
			with torch.no_grad():
				pred = int(self.policy_network(state).argmax())
		else:
			pred = random.randrange(len(EnumAction))

		if pred not in range(len(EnumAction)):
			return None
		return EnumAction(pred)

	@override
	def update_replay_memory(self, state, reward, action):
		if self.last_state is None:
			self.last_state = torch.zeros(state.shape).to(self.device)
		self.memory.push(Transition(self.last_state, action.value, state, reward))

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
		return get_model_lstm(device, input_dim, LSTM_SEQUENCE_SIZE, len(EnumAction), model_name)

	@override
	def select_action(self, state):
		sample = random.random()

		if len(self.sequence_queue) >= LSTM_SEQUENCE_SIZE and sample > self.eps:
			with torch.no_grad():
				prev_states = list(self.sequence_queue)[1:]
				prev_states.append(state)
				state_vector = torch.stack(prev_states)
				pred = int(self.policy_network(state_vector).argmax())
		else:
			pred = random.randrange(len(EnumAction))

		if pred not in range(len(EnumAction)):
			return None
		return EnumAction(pred)

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
		return get_model_cnn(device, input_dim, len(EnumAction), model_name)

	@override
	def select_action(self, state):
		sample = random.random()

		if sample > self.eps:
			with torch.no_grad():
				pred = int(self.policy_network(state).argmax())
		else:
			pred = random.randrange(len(EnumAction))

		if pred not in range(len(EnumAction)):
			return None
		return EnumAction(pred)

	@override
	def update_replay_memory(self, state, reward, action):
		if self.last_state is None:
			self.last_state = torch.zeros(state.shape).to(self.device)
		self.memory.push(Transition(self.last_state, action.value, state, reward))

	@override
	def post_step(self, state):
		self.last_state = state
