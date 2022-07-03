import unittest

from ai_wars.enums import EnumAction, AlwaysForwardsActions

class TestRotationOnlyActions(unittest.TestCase):

	def test_to_enum_action_left(self):
		expected = {EnumAction.FORWARD, EnumAction.LEFT}
		self.assertEqual(expected, AlwaysForwardsActions.LEFT_FORWARD.to_action_set())

	def test_to_enum_action_right(self):
		expected = {EnumAction.FORWARD, EnumAction.RIGHT}
		self.assertEqual(expected, AlwaysForwardsActions.RIGHT_FORWARD.to_action_set())
