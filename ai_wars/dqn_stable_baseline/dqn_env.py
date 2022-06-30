import logging
import time
import numpy as np
import gym
from gym import spaces

from ai_wars.networking.client import UdpClient
from ai_wars.networking.layers.compression import GzipCompression

from ai_wars.client.serializer import serialize_action
from ai_wars.client.deserializer import deserialize_game_state

from ai_wars.dqn.dqn_utils import gamestate_to_tensor

from ai_wars.constants import (
	CLIENT_BUFFER_SIZE,
	CLIENT_TIMEOUT,
	WIDTH,
	HEIGHT,
	MOVEMENT_SET
)

class ClientEnvironment(gym.Env):

	def __init__(self, name: str, addr="127.0.0.1", port=1337):
		self.name = name

		self.observation_space = spaces.Box(
			np.array([0, 0, -1.0, -1.0]),
			np.array([WIDTH, HEIGHT, 1.0, 1.0]),
			shape=(4,), dtype=float
		)
		self.action_space = spaces.Discrete(len(MOVEMENT_SET))

		self.client = UdpClient.builder() \
			.with_buffer_size(CLIENT_BUFFER_SIZE) \
			.with_timeout(CLIENT_TIMEOUT) \
			.add_layer(GzipCompression()) \
			.build()

		self.client.connect(addr, port)

		self.last_score = 0

	def step(self, action):
		data_out = serialize_action(self.name, MOVEMENT_SET(action).to_action_set())
		self.client.send(data_out.encode())

		players, scoreboard = self._read_next_gamestate()
		score = scoreboard[self.name].score
		if score > self.last_score:
			reward = 1
		elif score < self.last_score:
			reward = -1
		else:
			reward = 0
		gamestate_tensor = gamestate_to_tensor(self.name, players)

		return gamestate_tensor, reward, abs(score) > 10000, {}

	def reset(self):
		self.client.send(serialize_action(self.name, []).encode())
		players, _ = self._read_next_gamestate()
		return gamestate_to_tensor(self.name, players)

	def render(self, mode="human"):
		pass

	def close(self):
		pass

	def _read_next_gamestate(self):
		data_in = None
		while data_in is None:
			try:
				data_in = self.client.recv_next()
			except TimeoutError:
				logging.warning("Could not recieve data from server")
				time.sleep(0.1)
		return deserialize_game_state(data_in.decode())
