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
			[1.0, 1.0, 1.0, 1.0], # own
			[2.0, 2.0, 2.0, 2.0], # other1
			[0.0, 0.0, 0.0, 0.0], # other2 (empty)
			[1.0, 1.0, 1.0, 1.0], # projectile 1
			[2.0, 2.0, 2.0, 2.0], # projectile 2
			[3.0, 3.0, 3.0, 3.0], # projectile 3
			[4.0, 4.0, 4.0, 4.0], # projectile 4
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

		projectiles = [
			{
				"owner": "own",
				"position": Vector2(100.0, 100.0),
				"direction": Vector2(100.0, 100.0)
			}, {
				"owner": "other1",
				"position": Vector2(1.0, 1.0),
				"direction": Vector2(1.0, 1.0)
			}, {
				"owner": "other1",
				"position": Vector2(2.0, 2.0),
				"direction": Vector2(2.0, 2.0)
			}, {
				"owner": "other1",
				"position": Vector2(3.0, 3.0),
				"direction": Vector2(3.0, 3.0)
			}, {
				"owner": "other1",
				"position": Vector2(4.0, 4.0),
				"direction": Vector2(4.0, 4.0)
			}
		]

		actual = gamestate_to_tensor(
			"own",
			player,
			projectiles
		)

		self.assertTrue(expected.equal(actual))
