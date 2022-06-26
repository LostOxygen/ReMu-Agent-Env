import unittest

import torch
from pygame import Vector2

from ai_wars import constants
constants.NUM_PLAYERS = 2
constants.MAX_NUM_PROJECTILES = 4

from ai_wars.dqn.dqn_utils import gamestate_to_tensor

class TestDqnUtils(unittest.TestCase):

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
