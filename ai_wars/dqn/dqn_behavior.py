from ..enums import EnumAction

from ..client.behavior import Behavior
from ..utils import override, render_to_surface, surface_to_tensor, convert_to_greyscale

from .dqn_utils import gamestate_to_tensor
from .dqn_agent import DQNAgent

INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = 5
SEED = 0
GAMMA = 0.99           # discount factor
BUFFER_SIZE = 100000   # replay buffer size
BATCH_SIZE = 32        # Update batch size
LR = 0.0001            # learning rate
TAU = 1e-3             # for soft update of target parameters
UPDATE_EVERY = 100     # how often to update the network
UPDATE_TARGET = 10000  # After which thershold replay to be started
EPS_START = 0.99       # starting value of epsilon
EPS_END = 0.01         # Ending value of epsilon
EPS_DECAY = 100       # Rate by which epsilon to be decayed


class DqnBehavior(Behavior):
	"""DQN Behavior"""

	def __init__(self, player_name: str, agent_name: str, device="cpu"):
		self.player_name = player_name
		self.agent_name = agent_name

		self.device = device

		self.steps_done = 0
		self.running_loss = 0
		self.prev_state = None

		self.agent = None

		self.last_score = 0

	@override
	def make_move(self,
		players: dict[str, any],
		projectiles: dict[str, any],
		scoreboard: dict[str, int]
	) -> set[EnumAction]:
		# obtain the new score and calculate the reward
		new_score = scoreboard[self.player_name]
		reward = new_score - self.last_score

		# prepare the gamestate for the model
		if self.agent_name == "cnn":
			gamestate_surface = render_to_surface(players, projectiles)
			gamestate_tensor = surface_to_tensor(gamestate_surface, self.device)
			gamestate_tensor = convert_to_greyscale(gamestate_tensor)
		else:
			gamestate_tensor = gamestate_to_tensor(self.player_name, players, projectiles, self.device)
			gamestate_tensor = gamestate_tensor.flatten()

		# check if the model is already loaded, if not load it
		# if self.optimizer is None:
		# 	self.optimizer = get_agent(self.agent_name, self.device,
		# 							   self.player_name, len(gamestate_tensor))
		if self.prev_state is None:
			self.prev_state = gamestate_tensor

		if self.agent is None:
			self.agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, self.device, BUFFER_SIZE,
                         		  len(gamestate_tensor), self.agent_name, self.player_name,
								  BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET)

		# let the network predict (outputs an tensor with q-values for all actions)
		predicted_action = self.agent.act(gamestate_tensor)

		# run the next training step
		self.agent.step(self.prev_state, predicted_action, reward, gamestate_tensor)
		# self.steps_done += 1
		# self.running_loss += loss

		# print(f"loss: {(self.running_loss/self.steps_done):8.2f}\teps: {eps:8.2f} "\
		# 	  f"\tmax q value: {max_q_value:8.2f}\tsteps: {self.steps_done}", end="\r")

		# save the current state and actions for the next iteration
		self.last_score = new_score
		self.prev_state = gamestate_tensor

		if predicted_action is None:
			print(predicted_action, type(predicted_action))
			return {}

		# return the action enum with the highest q value as a set
		return {predicted_action}
