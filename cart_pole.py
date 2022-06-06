from enum import Enum
import gym
import torch
from torch import nn
from matplotlib import pyplot

class Actions(Enum):
	LEFT = 0
	RIGHT = 1

from ai_wars import constants
constants.MOVEMENT_SET = Actions
constants.BATCH_SIZE = 64
constants.MEMORY_SIZE = 100_000
constants.GAMMA = 0.99
constants.EPS_START = 1.0
constants.EPS_END = 0.5
constants.DECAY_FACTOR = 0.995

class DQNModelLinear(nn.Module):
	"""DQN Model with fully connected layers"""

	SIZE = 64

	def __init__(self, input_dim: int, num_actions: int) -> None:
		super().__init__()

		self.layers = nn.Sequential(
			nn.Linear(input_dim, self.SIZE),
			nn.ReLU(inplace=True),
			nn.Linear(self.SIZE, num_actions)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.layers(x)
		return x

def get_model_linear(device: str, input_dim: int, output_dim: int, _: str) -> torch.nn.Module:
	return DQNModelLinear(input_dim, output_dim).to(device)

from ai_wars.dqn import dqn_utils
dqn_utils.get_model_linear = get_model_linear

from ai_wars.dqn.dqn_agent import LinearAgent

ITERATION = 2000

if __name__ == "__main__":
	env = gym.make("CartPole-v1").unwrapped

	dqn = LinearAgent("cuda", "cart_pole", 4, episodes_per_update=50)

	steps = torch.zeros(ITERATION)
	for i in range(ITERATION):
		state = torch.tensor(env.reset(), device="cuda")
		done = False
		score = 0
		dqn.last_state = None

		while not done:
			action = dqn.select_action(state)
			state, reward, done, _ = env.step(action.value)
			state = torch.tensor(state, device="cuda")

			if done:
				reward = -100

			loss, eps, q_val = dqn.apply_training_step(state, reward, action)

			steps[i] += 1

		env.close()

		mean = torch.mean(steps[steps > 0])
		print(f"{i:5} steps: {steps[i]:4.0f}, mean: {mean:4.1f}, eps: {eps:3.2f}")

	pyplot.xlabel("Iterations")
	pyplot.ylabel("Steps")
	pyplot.plot(steps)
	pyplot.savefig("results.png")

	print("Testing")
	dqn.eps = 0.0
	for i in range(25):
		observation = env.reset()
		steps = 0

		done = False
		while not done:
			action = dqn.select_action(torch.tensor(observation, device="cuda"))
			observation, reward, done, info = env.step(action.value)
			env.render()
			steps += 1

		print(f"{i:3}: {steps:4} steps")
