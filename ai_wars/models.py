"""Library for neural network and general model architectures"""
from torch import nn
from torch import Tensor


class DQNModel(nn.Module):
	"""DQN Model"""

	def __init__(self, input_dim: int, num_actions: int) -> None: # pylint: disable=useless-super-delegation
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(input_dim, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, num_actions)
		)

	def forward(self, x: Tensor) -> Tensor:
		x = self.layers(x)
		return x

