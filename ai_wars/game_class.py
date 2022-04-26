"""Main GameClass"""
import sys
import pygame

from ai_wars.utils import load_sprite
from ai_wars.spaceship import Spaceship
from ai_wars.enums import EnumAction
from ai_wars.scoreboard import Scoreboard


class GameClass:
	"""MainGameClass"""

	# some constants
	# pygame userevents use codes from 24 to 35, so the first user event will be 24
	DECREASE_SCORE_EVENT = pygame.USEREVENT + 0 # event code 24
	SHOOT_COOLDOWN = 200 # specifies the cooldown for shooting in ms

	def __init__(self):

		pygame.init()
		self.screen = pygame.display.set_mode((800, 600))
		self.clock = pygame.time.Clock()
		self.background = load_sprite("ai_wars/img/space.png", False)

		# initialize the scoreboard and attach all players as observers
		self.scoreboard = Scoreboard()

		self.bullets = [] # list with all bullets in the game
		self.spaceships = [] # list with every spaceship in the game
		self.spaceship = Spaceship(400, 300, 40, 40, \
								   load_sprite("ai_wars/img/spaceship.png"), \
								   self.bullets.append, self.screen,
								   self.scoreboard, "Player 1")
		self.spaceship2 = Spaceship(400, 300, 40, 40,
                             load_sprite("ai_wars/img/spaceship.png"),
                             self.bullets.append, self.screen,
							 self.scoreboard, "Player 2")
		# append the spaceship to the list of spaceships, later the game will append the
		# spaceships of every player to this list
		self.spaceships.append(self.spaceship)
		self.spaceships.append(self.spaceship2)

		# attach all players to the scoreboard
		for ship in self.spaceships:
			self.scoreboard.attach(ship)

		# initialize timer and delta_time
		self.clock = pygame.time.Clock()
		self.delta_time = 0
		self.time_elapsed_since_last_action = 0

		# initialize custom event timer
		self.decrease_score_event = pygame.event.Event(self.DECREASE_SCORE_EVENT,
													   message="decrease score")
		pygame.time.set_timer(self.decrease_score_event, 1000)

	def main_loop(self) -> None:
		"""main loop for input handling, game logic and rendering"""
		while True:
			self.delta_time = self.clock.tick(144)
			self.time_elapsed_since_last_action += self.delta_time

			self._handle_inputs()
			self._handle_events()
			self._process_game_logic()
			self._draw()

	def _handle_inputs(self) -> None:
		"""private method to process inputs and limit the bullet frequency"""
		# check whick keys eare pressed
		is_key_pressed = pygame.key.get_pressed()

		match is_key_pressed:
			case is_key_pressed if is_key_pressed[pygame.K_SPACE]:
				# limit the frequency of bullets
				if self.time_elapsed_since_last_action > self.SHOOT_COOLDOWN:
					self.spaceship.action(EnumAction.SHOOT)
					self.time_elapsed_since_last_action = 0
			case is_key_pressed if is_key_pressed[pygame.K_LEFT]:
				self.spaceship.action(EnumAction.LEFT)
			case is_key_pressed if is_key_pressed[pygame.K_RIGHT]:
				self.spaceship.action(EnumAction.RIGHT)
			case is_key_pressed if is_key_pressed[pygame.K_UP]:
				self.spaceship.action(EnumAction.FORWARD)
			case is_key_pressed if is_key_pressed[pygame.K_DOWN]:
				self.spaceship.action(EnumAction.BACKWARD)

	def _handle_events(self) -> None:
		"""Private helper method to listen and process pygame events"""
		for event in pygame.event.get():
			match event:
				# check if the game should be closed
				case event if event.type == pygame.QUIT or \
					 (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
					sys.exit()

				# decrease the score of the players (event gets fired every second)
				case _ if event.type == pygame.USEREVENT+0:
					for ship in self.spaceships:
						self.scoreboard.decrease_score(ship.name, 1)

	def _draw(self) -> None:
		"""private method to draw the game"""

		# draw the background
		self.screen.blit(self.background, (0, 0))

		# draw the spaceship
		for spaceship in self.spaceships:
			spaceship.draw(self.screen)

		# rendering loop to draw all bullets
		for bullet in self.bullets:
			# if the bullet is out of the screen, let it despawn
			if bullet.x > self.screen.get_width() or \
			   bullet.x < 0 or \
			   bullet.y > self.screen.get_height() or \
			   bullet.y < 0:
				self.bullets.remove(bullet)
				del bullet
			else:
				bullet.draw(self.screen)

		# draw scoreboard
		self.scoreboard.draw_scoreboard(self.screen)

		pygame.display.flip()

	def _process_game_logic(self) -> None:
		"""private method to process game logic"""

		# loop over every bullet and update its position
		for bullet in self.bullets:
			bullet.move()
