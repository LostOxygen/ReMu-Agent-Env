"""Library for neural network and general model architectures"""
from torch import nn
from torch import Tensor


class DQNModel(nn.Module):
	"""DQN Model"""

	def __init__(self) -> None: # pylint: disable=useless-super-delegation
		super().__init__()
		self.layers = nn.Sequential(
			nn.Identity()
		)

	def forward(self, x: Tensor) -> Tensor:
		x = self.layers(x)
		return x
