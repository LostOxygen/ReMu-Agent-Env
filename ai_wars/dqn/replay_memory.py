import random
from collections import namedtuple, deque


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory:
	'''
	Finite queue of transitions.
	'''

	def __init__(self, capacity):
		self.memory = deque([], maxlen=capacity)

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

		return random.sample(self.memory, n)

	def __len__(self):
		return len(self.memory)
