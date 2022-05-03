import unittest

from ai_wars.enums import EnumAction

from ai_wars.server import deserializer

class TestSerializer(unittest.TestCase):

	# pylint: disable=protected-access

	def test__string_to_action_success(self):
		expected = EnumAction.SHOOT

		actual = deserializer._string_to_action("shoot")

		self.assertEqual(expected, actual)

	def test__string_to_action_fail(self):
		with self.assertRaises(ValueError):
			deserializer._string_to_action("kill_yourself")

	def test_deserialize_action(self):
		expected = ("Dieter", [EnumAction.SHOOT])

		json_string = \
			'''{
				"player_name": "Dieter",
				"action": [ "shoot" ]
			}'''
		actual = deserializer.deserialize_action(json_string)

		self.assertEqual(expected, actual)
