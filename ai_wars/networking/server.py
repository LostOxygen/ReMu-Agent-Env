import socket
from typing import Tuple

from .layers.layer import Layer
from .client_entry import ClientEntry

class UdpServer:
	'''
	Simple abstraction of a UDP server.
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
		Accessability method for server builder.

		Returns:
			server builder
		'''

		return UdpServerBuilder()

	def start(self, addr: str, port: int):
		'''
		Lets the server server on the given address and port

		Parameters:
			addr: address the server will bind to
			port: port the the server will listen on
		'''

		self.socket.bind((addr, port))
		print(f"started server on addr:{addr} - port {port}")

	def send_to_all(self, data: any):
		'''
		Sends the given bytes to all registered clients. Applies all layers before sending.

		Parameters:
			data: data that should be send
		'''

		for layer in self.layers:
			data = layer.forward(data)

		for client in map(ClientEntry.as_tuple, self.clients):
			self.socket.sendto(data, client)

	def recv_next(self) -> Tuple[ClientEntry, any]:
		'''
		Receive next incoming data. Applies all layers before returning.
		When data from a new client arrives, it is added to the known clients.

		Returns:
			(client that send the message, data that was received)
		'''

		data, addr = self.socket.recvfrom(self.buffer_size)

		self.clients.add(ClientEntry(*addr))

		for layer in self.layers:
			data = layer.backward(data)

		return data

class UdpServerBuilder:
	'''
	Builder for Server
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

	def build(self) -> UdpServer:
		'''
		Builds the server with configured parameters.

		Returns:
			the server object
		'''

		return UdpServer(self.buffer_size, self.layers)
