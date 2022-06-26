import unittest

from ai_wars.enums import EnumAction, RotationOnlyActions

class TestRotationOnlyActions(unittest.TestCase):

	def test_to_enum_action_left(self):
		self.assertEqual(EnumAction.LEFT, RotationOnlyActions.LEFT.to_enum_action())

	def test_to_enum_action_right(self):
		self.assertEqual(EnumAction.RIGHT, RotationOnlyActions.RIGHT.to_enum_action())
