"""Main GameClass"""
import sys
import pygame

from ai_wars.utils import load_sprite
from ai_wars.spaceship import Spaceship
from ai_wars.enums import EnumAction

DECREASE_SCORE_EVENT = pygame.USEREVENT + 0

class GameClass:
	"""MainGameClass"""
	def __init__(self):

		pygame.init()
		pygame.font.init()
		self.font = pygame.font.SysFont("consolas", 15)
		self.font_width = self.font.size("X")[0]
		self.font_height = self.font.size("X")[1]
		self.screen = pygame.display.set_mode((800, 600))
		self.clock = pygame.time.Clock()
		self.background = load_sprite("ai_wars/img/space.png", False)
		self.bullets = [] # list with all bullets in the game
		self.spaceships = [] # list with every spaceship in the game
		self.spaceship = Spaceship(400, 300, 40, 40, \
								   load_sprite("ai_wars/img/spaceship.png"), \
								   self.bullets.append, self.screen, "Player 1")
		# append the spaceship to the list of spaceships, later the game will append the
		# spaceships of every player to this list
		self.spaceships.append(self.spaceship)
		self.leaderboard = {}

		# initialize timer and delta_time
		self.clock = pygame.time.Clock()
		self.delta_time = 0
		self.time_elapsed_since_last_action = 0

		# initialize custom event timer
		self.decrease_score_event = pygame.event.Event(DECREASE_SCORE_EVENT, message="decrease score")
		pygame.time.set_timer(self.decrease_score_event, 1000)

	def main_loop(self) -> None:
		"""main loop for input handling, game logic and rendering"""
		while True:
			self.delta_time = self.clock.tick(144)
			self.time_elapsed_since_last_action += self.delta_time

			self._handle_events()
			self._process_game_logic()
			self._draw()

	def _handle_events(self) -> None:
		"""Private helper method to listen for and process input via
			pygame events
		"""
		for event in pygame.event.get():
			match event:
				# check if the game should be closed
				case event if event.type == pygame.QUIT or \
					 (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
					sys.exit()

				# decrease the score of the players (event gets fired every second)
				case _ if event.type == pygame.USEREVENT+0:
					for spaceship in self.spaceships:
						spaceship.score -= 1

		# check whick keys eare pressed
		is_key_pressed = pygame.key.get_pressed()

		match is_key_pressed:
			case is_key_pressed if is_key_pressed[pygame.K_SPACE]:
				self.spaceship.action(EnumAction.SHOOT)
			case is_key_pressed if is_key_pressed[pygame.K_LEFT]:
				self.spaceship.action(EnumAction.LEFT)
			case is_key_pressed if is_key_pressed[pygame.K_RIGHT]:
				self.spaceship.action(EnumAction.RIGHT)
			case is_key_pressed if is_key_pressed[pygame.K_UP]:
				self.spaceship.action(EnumAction.FORWARD)
			case is_key_pressed if is_key_pressed[pygame.K_DOWN]:
				self.spaceship.action(EnumAction.BACKWARD)

	def _draw_leaderboard(self, screen) -> None:
		"""private method to draw the leaderboard on the given screen"""
		for pos, (player, score) in enumerate(self.leaderboard.items()):
			score_string = f"{player} : {score}"
			text_surface = self.font.render(score_string, False, (255, 255, 255))
			screen.blit(text_surface, (0, self.font_height*pos))


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

		# draw leaderboard
		self._draw_leaderboard(self.screen)

		pygame.display.flip()

	def _process_game_logic(self) -> None:
		"""private method to process game logic"""

		# loop over every bullet and update its position
		for bullet in self.bullets:
			bullet.move()

		# update the leaderboard
		for ship in self.spaceships:
			self.leaderboard[ship.name] = ship.score
			self.leaderboard["Player 1337"] = 69
			self.leaderboard["Player 69"] = 42
			self.leaderboard["Player 42"] = 1337
		self.leaderboard = dict(sorted(self.leaderboard.items(), key=lambda x: x[1], reverse=False))
