from typing import Tuple
import random
from collections import namedtuple
import torch

Transition = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])

class ReplayMemory:
	"""Fixed-size buffer to store experience tuples."""

	def __init__(self, buffer_size: int, batch_size: int, device: str):
		"""
		Initialize a ReplayBuffer object.

		Parameters:
			buffer_size (int): maximum size of buffer
			batch_size (int): size of each training batch
			device (string): GPU or CPU
		"""

		self.capacity = buffer_size
		self.memory = []
		self.batch_size = batch_size
		self.experience = namedtuple("Experience",
									 field_names=["state", "action", "reward", "next_state"])
		self.device = device
		self.position = 0

	def add(self, state, action, reward, next_state):
		"""Add a new experience to memory."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = self.experience(
			state, action, reward, next_state)
		self.position = (self.position + 1) % self.capacity

	def sample(self) -> Tuple:
		"""Randomly sample a batch of experiences from memory."""
		experiences = random.sample(self.memory, k=self.batch_size)

		states = torch.stack([e.state for e in experiences if e is not None]).to(self.device)
		actions = torch.tensor([e.action for e in experiences if e is not None]).to(self.device)
		rewards = torch.tensor([e.reward for e in experiences if e is not None]).to(self.device)
		next_states = torch.stack([e.next_state for e in experiences if e is not None]).to(self.device)

		return (states, actions, rewards, next_states)

	def __len__(self):
		"""Return the current size of internal memory."""
		return len(self.memory)
