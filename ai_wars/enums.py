"""enum class for several enum implementations"""
from enum import Enum


class EnumAction(Enum):
	"""Enum class for different player actions"""
	LEFT = 0
	FORWARD = 1
	RIGHT = 2
	BACKWARDS = 3
	SHOOT = 4
