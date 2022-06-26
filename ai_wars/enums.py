"""enum class for several enum implementations"""
from enum import Enum

class MoveSet(Enum):
	"""
	Base enum for a movement set.
	"""

	def to_enum_action(self):
		"""
		Maps itself to an EnumAction that contains all possible actions.

		Returns:
			an EnumAction value
		"""
		pass

class EnumAction(MoveSet):
	"""Enum class for different player actions"""

	LEFT = 0
	RIGHT = 1
	FORWARD = 2
	BACKWARD = 3

	def to_enum_action(self):
		return self

	def __int__(self):
		return self.value

class RotationOnlyActions(MoveSet):
	"""Movement set that contains only rotating and shooting."""

	LEFT = 0
	RIGHT = 1

	def to_enum_action(self):
		match self:
			case RotationOnlyActions.LEFT:
				return EnumAction.LEFT
			case RotationOnlyActions.RIGHT:
				return EnumAction.RIGHT
