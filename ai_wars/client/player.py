import logging
from .behavior import Behavior
from ..networking.client import UdpClient
from ..networking.layers.compression import GzipCompression
from .deserializer import deserialize_game_state
from .serializer import serialize_action

from ..constants import CLIENT_BUFFER_SIZE

class Player:
	'''
	Player of the game that performs a generic action after each game update.
	'''

	def __init__(self, name: str, addr: str, port: int, behavior: Behavior):
		self.name = name
		self.addr = addr
		self.port = port
		self.behavior = behavior
		logging.debug("Initialized player on addr: %s and port: %s", self.addr, self.port)

	def change_behavior(self, new_behavior: Behavior):
		'''
		Changes the behavior of the player.

		Parametes:
			new_behavior: the new behavior
		'''

		self.behavior = new_behavior

	def loop(self):
		'''
		Connects the player to the server and let it play the given with its configured behavior.
		'''

		client = UdpClient.builder() \
			.with_buffer_size(CLIENT_BUFFER_SIZE) \
			.add_layer(GzipCompression()) \
			.build()

		client.connect(self.addr, self.port)
		client.send(serialize_action(self.name, []).encode())

		while True:
			data_in = client.recv_next()
			game_state = deserialize_game_state(data_in.decode())

			actions = self.behavior.make_move(*game_state)

			data_out = serialize_action(self.name, actions)
			client.send(data_out.encode())

class PlayerFactory:
	'''
	Factory for a Player for a single server.
	'''

	def __init__(self, addr: str, port: int):
		self.addr = addr
		self.port = port

	def create_player(self, name: str, behavior: Behavior) -> Player:
		'''
		Creates a new player with given name and behavior

		Parameters:
			name: name of the player
			behavior: behavior of the player

		Returns:
			a new player instance
		'''
		logging.debug("Created new player with name: %s", name)
		return Player(self.addr, self.port, name, behavior)
