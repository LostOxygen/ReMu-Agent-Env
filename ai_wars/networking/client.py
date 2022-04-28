import socket

from .layers.layer import Layer

class Client:
	'''
	Simple abstraction of a UDP client.
	'''

	def __init__(self,
		buffer_size: int,
		layers: list[Layer]
	):
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.buffer_size = buffer_size
		self.layers = layers
		self.clients = set()

	@staticmethod
	def builder():
		'''
		Accessability method for client builder.

		Returns:
			client builder
		'''

		return ClientBuilder()

	def connect(self, addr: str, port: int):
		self.socket.connect((addr, port))

	def send(self, data: any):
		'''
		Sends the given bytes to all registered clients. Applies all layers before sending.

		Parameters:
			data: data that should be send
		'''

		for layer in self.layers:
			data = layer.forward(data)

		self.socket.sendall(data)

	def recv_next(self) -> any:
		'''
		Receive next incoming data. Applies all layers before returning.
		When data from a new client arrives, it is added to the known clients.

		Returns:
			(client that send the message, data that was received)
		'''

		data, _ = self.socket.recvfrom(self.buffer_size)

		for layer in self.layers:
			data = layer.backward(data)

		return data

class ClientBuilder:
	'''
	Builder for client
	'''

	def __init__(self):
		self.buffer_size = 1204
		self.layers = []

	def with_buffer_size(self, size: int):
		self.buffer_size = size
		return self

	def add_layer(self, layer: Layer):
		'''
		Adds a layer that is applied when sending or receiving data. Calling order of the method
		declares the application of the layers.

		Parameters:
			layer: layer that should be added
		'''

		self.layers.append(layer)
		return self

	def build(self) -> Client:
		'''
		Builds the client with configured parameters.

		Returns:
			the client object
		'''

		return Client(self.buffer_size, self.layers)
