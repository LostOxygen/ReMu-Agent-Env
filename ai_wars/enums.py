"""enum class for several enum implementations"""
from enum import Enum

class MoveSet(Enum):
	"""
	Base enum for a movement set.
	"""

	def to_action_set(self) -> set:
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

	def to_action_set(self):
		return {self}

	def __int__(self):
		return self.value

class AlwaysForwardsActions(MoveSet):
	"""Movement set that contains only rotating and shooting."""

	FORWARD = 0
	RIGHT_FORWARD = 1
	LEFT_FORWARD = 2

	def to_action_set(self):
		match self:
			case AlwaysForwardsActions.LEFT_FORWARD:
				return {EnumAction.FORWARD, EnumAction.LEFT}
			case AlwaysForwardsActions.RIGHT_FORWARD:
				return {EnumAction.FORWARD, EnumAction.RIGHT}
			case AlwaysForwardsActions.FORWARD:
				return {EnumAction.FORWARD}
