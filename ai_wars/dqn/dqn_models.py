"""Library for neural network and general model architectures"""
from torch import nn
from torch import Tensor

class DQNModelLinear(nn.Module):
	"""DQN Model with fully connected layers"""

	SIZE = 64

	def __init__(self, input_dim: int, num_actions: int) -> None: # pylint: disable=useless-super-delegation
		super().__init__()

		self.layers = nn.Sequential(
			nn.Linear(input_dim, self.SIZE),
			nn.ReLU(inplace=True),
			nn.Linear(self.SIZE, num_actions),
			nn.Softmax(dim=-1)
		)

	def forward(self, x: Tensor) -> Tensor:
		x = self.layers(x)
		return x


class DQNModelCNN(nn.Module):
	"""DQN Model with conv's for feature extraction and fully connected layers for action mapping"""

	def __init__(self, input_dim: int, num_actions: int) -> None:  # pylint: disable=useless-super-delegation
		super().__init__()

		self.features = nn.Sequential(
					nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
					nn.ReLU(),
					nn.Conv2d(32, 64, kernel_size=4, stride=2),
					nn.ReLU(),
					nn.Conv2d(64, 64, kernel_size=3, stride=1),
					nn.ReLU()
				)

		self.fc = nn.Sequential(
                    nn.Linear(64*3, 512),
					nn.ReLU(),
					nn.Linear(512, num_actions)
				)

	def forward(self, x: Tensor) -> Tensor:
		x = self.features(x)
		x = x.flatten()
		x = self.fc(x)
		return x


class DQNModelLSTM(nn.Module):
	"""DQN Model with LSTM layer"""

	def __init__(self, num_features: int, sequence_length: int, num_actions: int) -> None: # pylint: disable=useless-super-delegation
		super().__init__()

		self.sequence_length = sequence_length

		self.layer2 = nn.LSTM(num_features, 16)
		self.layer3 = nn.Linear(sequence_length * 16, num_actions)

	def forward(self, x: Tensor) -> Tensor:
		batch_size = x.shape[0] if x.dim() == 3 else 1
		x, _ = self.layer2(x)
		x = x.view(batch_size, self.sequence_length*16)
		x = self.layer3(x)
		return x

