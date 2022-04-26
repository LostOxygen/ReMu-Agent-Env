import unittest

import json

from pygame.math import Vector2
from ai_wars.spaceship import Spaceship
from ai_wars.bullet import Bullet

from ai_wars.api import serializer_server

class TestSerializer(unittest.TestCase):

	# pylint: disable=protected-access

	def test__vector_as_dict(self):
		expected = {"x": 100.0, "y": 200.0}

		vec = Vector2(100.0, 200.0)
		actual = serializer_server._vector_as_dict(vec)

		self.assertEqual(expected, actual)

	def test__spaceship_as_dict(self):
		expected = {
			"name": "Dieter",
			"position": {"x": 5.0, "y": 6.0},
			"direction": "todo"
		}

		ship = Spaceship(5.0, 6.0, 100, 100, None, None, None, "Dieter")
		actual = serializer_server._spaceship_as_dict(ship)

		self.assertEqual(expected, actual)

	def test__bullet_as_dict(self):
		expected = {
			"owner": "Dieter",
			"position": {"x": 5.0, "y": 6.0},
			"direction": {"x": -1.0, "y": -2.0}
		}

		bullet = Bullet("Dieter", 5.0, 6.0, 100, 100, None, Vector2(-1.0, -2.0))
		actual = serializer_server._bullet_as_dict(bullet)

		self.assertEqual(expected, actual)

	def test__scoreboard_as_dict(self):
		expected = [
			{
				"name": "Dieter",
				"score": 100
			},
			{
				"name": "Bernd",
				"score": -50
			}
		]

		scoreboard = {
			"Dieter": 100,
			"Bernd": -50
		}
		actual = serializer_server._scoreboard_as_dict(scoreboard)

		self.assertEqual(expected, actual)

	def test_serialize_game_state(self):
		expected = json.dumps({
			"players": [
				{
					"name": "Dieter",
					"position": {"x": 5.0, "y": 6.0},
					"direction": "todo"
				},
				{
					"name": "Bernd",
					"position": {"x": 2.0, "y": 8.0},
					"direction": "todo"
				}
			],
			"projectiles": [
				{
					"owner": "Dieter",
					"position": {"x": 5.0, "y": 6.0},
					"direction": {"x": -1.0, "y": -2.0}
				},
				{
					"owner": "Dieter",
					"position": {"x": 8.0, "y": 8.0},
					"direction": {"x": -5.0, "y": -2.0}
				}
			],
			"scores": [
				{
					"name": "Dieter",
					"score": 100
				},
				{
					"name": "Bernd",
					"score": -50
				}
			]
		})

		spaceships = [
			Spaceship(5.0, 6.0, 100, 100, None, None, None, "Dieter"),
			Spaceship(2.0, 8.0, 100, 100, None, None, None, "Bernd")
		]
		bullets = [
			Bullet("Dieter", 5.0, 6.0, 100, 100, None, Vector2(-1.0, -2.0)),
			Bullet("Dieter", 8.0, 8.0, 100, 100, None, Vector2(-5.0, -2.0)),
		]
		scoreboard = {
			"Dieter": 100,
			"Bernd": -50
		}
		actual = serializer_server.serialize_game_state(spaceships, bullets, scoreboard)

		self.assertEqual(expected, actual)