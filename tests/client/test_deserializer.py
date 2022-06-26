import unittest

from pygame.math import Vector2

from ai_wars.client import deserializer

class TestDeserializerClient(unittest.TestCase):

	# pylint: disable=protected-access

	def test__dict_to_vector(self):
		expected = Vector2(5.0, 1.5)

		vector = {"x": 5.0, "y": 1.5}
		actual = deserializer._dict_to_vector(vector)

		self.assertEqual(expected, actual)

	def test__dict_to_player(self):
		expected = {
			"player_name": "Dieter",
			"position": Vector2(1.5, 5.2),
			"direction": Vector2(123.5, 123.2)
		}

		player = {
			"name": "Dieter",
			"position": {"x": 1.5, "y": 5.2},
			"direction": {"x": 123.5, "y": 123.2}
		}
		actual = deserializer._dict_to_player(player)

		self.assertEqual(expected, actual)

	def test__dict_as_scoreboard(self):
		expected = {"Dieter": 100}

		scoreboard = [{
			"name": "Dieter",
			"score": 100
		}]
		actual = deserializer._dict_as_scoreboard(scoreboard)

		self.assertEqual(expected, actual)

	def test_deserialize_action(self):
		expected = (
			[{
				"player_name": "Dieter",
				"position": Vector2(1.5, 5.2),
				"direction": Vector2(123.5, 123.2)
			}],
			{
				"Dieter": 100
			}
		)

		json_string = \
			'''{
				"players": [{
					"name": "Dieter",
					"position": {"x": 1.5, "y": 5.2},
					"direction": {"x": 123.5, "y": 123.2}
				}],
				"scoreboard": [{
					"name": "Dieter",
					"score": 100
				}]
			}'''
		actual = deserializer.deserialize_game_state(json_string)

		self.assertEqual(expected, actual)
