import unittest

import torch
from pygame import Vector2
from pygame import Rect

from ai_wars.maps.map import Map
from ai_wars.dqn.dqn_utils import gamestate_to_tensor, raycast_scan

from ai_wars import constants
constants.NUM_PLAYERS = 2
constants.MAX_NUM_PROJECTILES = 4


class TestMap(Map):

	def __init__(self):
		super().__init__(None)

		self.bound_rects.append(Rect(0, 0, 100, 20))
		self.bound_rects.append(Rect(0, 80, 100, 20))
		self.bound_rects.append(Rect(0, 0, 20, 100))
		self.bound_rects.append(Rect(80, 0, 20, 100))

class TestDqnUtils(unittest.TestCase):

	game_map = TestMap()

	def test_gamestate_to_tensor(self):
		expected = torch.tensor([
			[1.0, 1.0, 1.0, 1.0] # own
		])

		player = [
			{
				"player_name": "own",
				"direction": Vector2(1.0, 1.0),
				"position": Vector2(1.0, 1.0)
			},
			{
				"player_name": "other1",
				"direction": Vector2(2.0, 2.0),
				"position": Vector2(2.0, 2.0)
			}
		]

		actual = gamestate_to_tensor(
			"own",
			player
		)

		self.assertTrue(expected.equal(actual))

	def test_raycast_scan_single(self):
		expected = torch.tensor([30], dtype=torch.float32)

		origin = Vector2(50, 50)
		actual = raycast_scan(origin, self.game_map, num_rays=1, step_size=1)

		self.assertEqual(expected, actual)

	def test_raycast_scan_four(self):
		expected = torch.tensor([30, 30, 30, 30], dtype=torch.float32)

		origin = Vector2(50, 50)
		actual = raycast_scan(origin, self.game_map, num_rays=4, step_size=1)

		self.assertTrue(expected.equal(actual))

	def test_raycast_scan_eight(self):
		expected = torch.tensor([30, 43, 30, 43, 30, 43, 30, 43], dtype=torch.float32)

		origin = Vector2(50, 50)
		actual = raycast_scan(origin, self.game_map, num_rays=8, step_size=1)

		self.assertTrue(expected.equal(actual))
