import unittest

import json

from ai_wars.enums import EnumAction

from ai_wars.api import serializer_client

class TestSerializerClient(unittest.TestCase):

	# pylint: disable=protected-access

	def test__enum_action_as_dict(self):
		expected = ["left", "forward"]

		enum_actions = [EnumAction.LEFT, EnumAction.FORWARD]
		actual = serializer_client._enum_action_as_dict(enum_actions)

		self.assertEqual(expected, actual)

	def test_serialize_action(self):
		expected = json.dumps({
			"name": "Dieter",
			"actions": ["left", "forward"]
		})

		player_name = "Dieter"
		enum_actions = [EnumAction.LEFT, EnumAction.FORWARD]
		actual = serializer_client.serialize_action(player_name, enum_actions)

		self.assertEqual(expected, actual)
