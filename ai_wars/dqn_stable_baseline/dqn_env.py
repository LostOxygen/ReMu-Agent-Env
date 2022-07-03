import logging
import time
import gym
from gym import spaces
from pygame import Vector2

from ai_wars.networking.client import UdpClient
from ai_wars.networking.layers.compression import GzipCompression

from ai_wars.client.serializer import serialize_action
from ai_wars.client.deserializer import deserialize_game_state

from ai_wars.maps.map import Map

from ai_wars.dqn.dqn_utils import raycast_scan

from ai_wars.constants import (
	CLIENT_BUFFER_SIZE,
	CLIENT_TIMEOUT,
	MOVEMENT_SET
)


UP = Vector2(0, -1)

class ClientEnvironment(gym.Env):

	def __init__(self, name: str, game_map: Map, addr="127.0.0.1", port=1337):
		self.name = name
		self.map = game_map

		self.observation_space = spaces.Box(0, 1000, shape=(8,), dtype=float)
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
			reward = 1000
		elif score < self.last_score:
			reward = -1000
		else:
			reward = 0

		player = list(filter(lambda p: p["player_name"] == self.name, players))[0]
		player_pos = player["position"]
		player_angle = player["direction"].angle_to(UP)
		scan = raycast_scan(player_pos, player_angle, self.map)

		min_val = scan.min()
		if min_val > 0:
			reward += int(min_val)

		return scan, reward, abs(score) > 10000, {}

	def reset(self):
		self.client.send(serialize_action(self.name, []).encode())
		players, _ = self._read_next_gamestate()

		player = list(filter(lambda p: p["player_name"] == self.name, players))[0]
		player_pos = player["position"]
		player_angle = player["direction"].angle_to(UP)
		return raycast_scan(player_pos, player_angle, self.map)

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
