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

	def __init__(self, _: int, num_actions: int) -> None:  # pylint: disable=useless-super-delegation
		super().__init__()

		self.features = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                   	nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1),
                    nn.ReLU(inplace=True),
					nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
					nn.ReLU(inplace=True)
				)

		self.fc = nn.Sequential(
                    nn.Linear(32*74*99, 512),
               		nn.ReLU(inplace=True),
					nn.Linear(512, num_actions),
					nn.Softmax(dim=-1)
				)

	def forward(self, x: Tensor) -> Tensor:
		x = self.features(x)
		x = x.view(x.size(0), -1) # flatten with batch dimension
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

