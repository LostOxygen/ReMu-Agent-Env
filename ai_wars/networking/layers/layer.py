import abc

class Layer(abc.ABC):
	'''
	Isomorphism that may be applied before sending or receiving data
	'''

	@abc.abstractmethod
	def forward(self, data):
		'''
		Apply the forward step of isomorphism.

		Parameters:
			data: the data that should be used

		Returns:
			data after applying operation
		'''

		pass

	@abc.abstractmethod
	def backward(self, data):
		'''
		Apply the backward step of isomorphism.

		Parameters:
			data: the data that should be used

		Returns:
			data after applying operation
		'''

		pass
