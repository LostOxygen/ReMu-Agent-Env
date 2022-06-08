from typing import Tuple
import random
from collections import namedtuple, deque
import torch

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


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
		self.memory = deque([], maxlen=capacity)
		self.batch_size = batch_size
		self.device = device

	def push(self, transition: Transition):
		"""
		Push a transition to the queue.

		Parameters:
			transition: a transition
		"""

		self.memory.append(transition)

	def sample(self) -> list[Transition]:
		"""
		Samples BATCH_SZE random transition from the memory.

		Returns:
			BATCH_SIZE transitions
		"""

		experiences = random.sample(self.memory, k=self.batch_size)
		states = torch.stack([e.state for e in experiences]).to(self.device)
		actions = torch.tensor([e.action for e in experiences]).to(self.device)
		rewards = torch.tensor([e.reward for e in experiences]).to(self.device)
		next_states = torch.stack([e.next_state for e in experiences]).to(self.device)

		return (states, actions, rewards, next_states)

	def __len__(self):
		"""Return the current size of internal memory."""
		return len(self.memory)
