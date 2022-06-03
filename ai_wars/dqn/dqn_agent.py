import torch
import torch.nn.functional as F
from torch import optim
import random

from .replay_memory import ReplayMemory
from .dqn_utils import get_model_linear, get_model_cnn, get_model_lstm
from ..enums import EnumAction
from ..constants import (LSTM_SEQUENCE_SIZE)

class DQNAgent():
	"""DQN agent class"""
	def __init__(self, input_shape, action_size, device, buffer_size, input_dim, model_name,
				 player_name, batch_size, gamma, lr, tau, update_every, replay_after):
		"""Initialize an Agent object.

		Params
		======
			input_shape (tuple): dimension of each state (C, H, W)
			action_size (int): dimension of each action
			device(string): Use Gpu or CPU
			buffer_size (int): replay buffer size
			batch_size (int):  minibatch size
			gamma (float): discount factor
			lr (float): learning rate
			update_every (int): how often to update the network
			replay_after (int): After which replay to be started
			model(Model): Pytorch Model
		"""
		self.input_shape = input_shape
		self.action_size = action_size
		self.device = device
		self.buffer_size = buffer_size
		self.batch_size = batch_size
		self.gamma = gamma
		self.lr = lr
		self.update_every = update_every
		self.replay_after = replay_after
		match model_name:
			case "linear": self.model = get_model_linear(device, input_dim, 5, player_name)
			case "cnn":	self.model = get_model_cnn(device, input_dim, 5, player_name)
			case "lstm": self.model = get_model_lstm(device, input_dim, LSTM_SEQUENCE_SIZE, 5, player_name)

		self.tau = tau

		self.running_loss = 0.0
		self.eps = 0.99

		# Q-Network
		self.policy_net = self.model.to(self.device)
		self.target_net = self.model.to(self.device)
		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

		# Replay memory
		self.memory = ReplayMemory(self.buffer_size, self.batch_size, self.device)

		self.t_step = 0
		self.total_steps = 0


	def step(self, state, action, reward, next_state):
		self.total_steps += 1
		# Save experience in replay memory
		self.memory.add(state, action.value, reward, next_state)

		# Learn every UPDATE_EVERY time steps.
		self.t_step = (self.t_step + 1) % self.update_every

		if self.t_step == 0:
			# If enough samples are available in memory, get random subset and learn
			if len(self.memory) > self.replay_after:
				experiences = self.memory.sample()
				self.learn(experiences)


	def act(self, state):
		"""Returns actions for given state as per current policy."""
		self.eps = max(0.01, 0.999995 * self.eps)

		state = state.unsqueeze(0).to(self.device)
		self.policy_net.eval()

		# Epsilon-greedy action selection
		if random.random() > self.eps:
			with torch.no_grad():
				pred = int(self.policy_net(state).argmax())
		else:
			pred = random.randrange(len(EnumAction))

		self.policy_net.train()
		if pred not in range(len(EnumAction)):
			return None
		return EnumAction(pred)


	def learn(self, experiences):
		states, actions, rewards, next_states = experiences

		# Get expected Q values from policy model
		q_expected_current = self.policy_net(states)
		q_expected = q_expected_current.gather(
			1, actions.unsqueeze(1)).squeeze(1)

		# Get max predicted Q values (for next states) from target model
		q_targets_next = self.target_net(next_states).detach().max(1)[0]

		# Compute Q targets for current states
		q_targets = rewards + (self.gamma * q_targets_next)

		# Compute loss
		loss = F.mse_loss(q_expected, q_targets)

		# Minimize the loss
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.running_loss += loss.item()
		print(f"loss: {(self.running_loss/self.total_steps):8.2f}\teps: {self.eps:8.2f} "
              f"\tmax q value: {q_targets_next.max().item():8.2f}\tsteps: {self.total_steps}", end="\r")

		self.soft_update(self.policy_net, self.target_net)


	# θ'=θ×τ+θ'×(1−τ)
	def soft_update(self, policy_model, target_model):
		for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
			target_param.data.copy_(self.tau*policy_param.data + (1.0-self.tau)*target_param.data)
