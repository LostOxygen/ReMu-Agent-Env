import random
from collections import namedtuple, deque
import torch

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory:
	'''
	Finite queue of transitions.
	'''

	def __init__(self, capacity):
		self.memory = deque([], maxlen=capacity)
		self.device = "cuda"

	def push(self, transition: Transition):
		'''
		Push a transition to the queue.

		Parameters:
			transition: a transition
		'''

		self.memory.append(transition)

	def sample(self, n) -> list[Transition]:
		'''
		Samples n random transition from the memory.

		Returns:
			n transitions
		'''

		experiences = random.sample(self.memory, n)
		states = torch.stack([e.state for e in experiences]).to(self.device)
		actions = torch.tensor([e.action for e in experiences]).to(self.device)
		rewards = torch.tensor([e.reward for e in experiences]).to(self.device)
		next_states = torch.stack([e.next_state for e in experiences]).to(self.device)

		return (states, actions, rewards, next_states)

	def __len__(self):
		return len(self.memory)
