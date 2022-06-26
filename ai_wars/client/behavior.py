import abc

from ..enums import EnumAction

class Behavior(abc.ABC):
	'''
	Expresses a particular behavior for a player. This may be user input or a computer player like
	for example a neural network.
	'''

	@abc.abstractmethod
	def make_move(self,
		players: dict[str, any],
		scoreboard: dict[str, int]
	) -> set[EnumAction]:
		'''
		Inquires the actions the players performs given current game information.
		See `deserializer#deserialize_game_state`

		Parameters:
			players: the position and direction of each player.
			scoreboard: the score of each player

		Returns:
			a list of actions the player performs
		'''
		pass
