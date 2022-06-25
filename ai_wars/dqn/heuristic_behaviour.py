from ..enums import EnumAction

from ..client.behavior import Behavior
from ..utils import override

from .heuristic_agent import get_agent


class HeuristicBehavior(Behavior):
	"""Heuristic Agent Behavior"""

	def __init__(self, player_name: str, agent_name: str, device: str="cpu"):
		self.player_name = player_name
		self.agent_name = agent_name
		self.device = device

	@override
	def make_move(self,
			   players: list[dict[str, any]],
			   projectiles: list[dict[str, any]],
			   scoreboard: dict[str, int]
			   ) -> set[EnumAction]:

		agent = get_agent(self.agent_name, self.device, self.player_name)
		next_action = agent.select_action(players, projectiles)

		# return the action enum with the highest q value as a set
		return {next_action.to_enum_action()}
