import unittest

import json

from pygame import Rect
from pygame.sprite import Sprite
from pygame.math import Vector2
from ai_wars.scoreboard import ScoreboardEntry
from ai_wars.spaceship import Spaceship
from ai_wars.constants import COLOR_ARRAY

from ai_wars.server import serializer

class DummySprite(Sprite):

	def get_rect(self):
		return Rect(0, 0, 100, 100)

	def copy(self):
		return DummySprite()

	def fill(self, color, special_flags=0):
		pass

class TestSerializer(unittest.TestCase):

	# pylint: disable=protected-access

	def test__vector_as_dict(self):
		expected = {"x": 100.0, "y": 200.0}

		vec = Vector2(100.0, 200.0)
		actual = serializer._vector_as_dict(vec)

		self.assertEqual(expected, actual)

	def test__spaceship_as_dict(self):
		expected = {
			"name": "Dieter",
			"position": {"x": 5.0, "y": 6.0},
			"direction": {"x": 1.5, "y": 0.1}
		}

		ship = Spaceship(5.0, 6.0, DummySprite(), None, "Dieter", COLOR_ARRAY[0])
		ship.direction = Vector2(1.5, 0.1)
		actual = serializer._spaceship_as_dict(ship)

		self.assertEqual(expected, actual)

	def test__scoreboard_as_dict(self):
		expected = [
			{
				"name": "Dieter",
				"score": 100,
				"finish_reached": 1
			},
			{
				"name": "Bernd",
				"score": -50,
				"finish_reached": 4
			}
		]

		scoreboard = {
			"Dieter": ScoreboardEntry(100, 1),
			"Bernd": ScoreboardEntry(-50, 4)
		}
		actual = serializer._scoreboard_as_dict(scoreboard)

		self.assertEqual(expected, actual)

	def test_serialize_game_state(self):
		expected = json.dumps({
			"players": [
				{
					"name": "Dieter",
					"position": {"x": 5.0, "y": 6.0},
					"direction": {"x": 1.5, "y": 0.1}
				},
				{
					"name": "Bernd",
					"position": {"x": 2.0, "y": 8.0},
					"direction": {"x": 0.5, "y": 12.0}
				}
			],
			"scoreboard": [
				{
					"name": "Dieter",
					"score": 100,
					"finish_reached": 1
				},
				{
					"name": "Bernd",
					"score": -50,
					"finish_reached": 4
				}
			]
		})

		ship_dieter = Spaceship(5.0, 6.0, DummySprite(), None, "Dieter", COLOR_ARRAY[0])
		ship_dieter.direction = Vector2(1.5, 0.1)
		ship_bernd = Spaceship(2.0, 8.0, DummySprite(), None, "Bernd", COLOR_ARRAY[0])
		ship_bernd.direction = Vector2(0.5, 12.0)
		spaceships = [ship_dieter, ship_bernd]
		scoreboard = {
			"Dieter": ScoreboardEntry(100, 1),
			"Bernd": ScoreboardEntry(-50, 4)
		}
		actual = serializer.serialize_game_state(spaceships, scoreboard)

		self.assertEqual(expected, actual)
