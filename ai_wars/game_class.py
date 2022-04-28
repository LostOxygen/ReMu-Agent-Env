"""Main GameClass"""
import sys
import pygame
from typing import Callable

from ai_wars.utils import load_sprite
from ai_wars.spaceship import Spaceship
from ai_wars.enums import EnumAction
from ai_wars.scoreboard import Scoreboard
from ai_wars.bullet import Bullet

DECREASE_SCORE_EVENT = pygame.USEREVENT + 0
FRAMERATE = 144


class GameClass:
	"""MainGameClass"""

	# some constants
	# pygame userevents use codes from 24 to 35, so the first user event will be 24
	DECREASE_SCORE_EVENT = pygame.USEREVENT + 0 # event code 24
	POINTS_LOST_AFTER_GETTING_HIT = 100
	POINTS_GAINED_AFTER_HITTING = 200

	def __init__(self):

		pygame.init()
		self.screen = pygame.display.set_mode((800, 600))
		self.background = load_sprite("ai_wars/img/space.png", False)
		# initialize the scoreboard and attach all players as observers
		self.scoreboard = Scoreboard()
		self.bullets = [] # list with all bullets in the game
		self.spaceships = [] # list with every spaceship in the game
		self.spaceship1 = self.spawn_spaceship(100, 300, load_sprite("ai_wars/img/spaceship.png"),\
												self.bullets.append, self.screen, "Player 1")
		self.spaceship2 = self.spawn_spaceship(400, 300, load_sprite("ai_wars/img/spaceship.png"),\
												self.bullets.append, self.screen, "Player 2")
		# initialize custom event timer
		self.decrease_score_event = pygame.event.Event(self.DECREASE_SCORE_EVENT,
													   message="decrease score")
		pygame.time.set_timer(self.decrease_score_event, 1000)

	def main_loop(self) -> None:
		"""main loop for input handling, game logic and rendering"""
		while True:
			self._handle_inputs()
			self._handle_events()
			self._process_game_logic()
			self._draw()

	def _handle_inputs(self) -> None:
		"""private method to process inputs and limit the bullet frequency"""
		# check which keys are pressed
		is_key_pressed = pygame.key.get_pressed()

		match is_key_pressed:
			case is_key_pressed if is_key_pressed[pygame.K_SPACE]:
				self.spaceship1.action(EnumAction.SHOOT)
			case is_key_pressed if is_key_pressed[pygame.K_LEFT]:
				self.spaceship1.action(EnumAction.LEFT)
			case is_key_pressed if is_key_pressed[pygame.K_RIGHT]:
				self.spaceship1.action(EnumAction.RIGHT)
			case is_key_pressed if is_key_pressed[pygame.K_UP]:
				self.spaceship1.action(EnumAction.FORWARD)
			case is_key_pressed if is_key_pressed[pygame.K_DOWN]:
				self.spaceship1.action(EnumAction.BACKWARD)

	def _handle_events(self) -> None:
		"""Private helper method to listen and process pygame events"""
		for event in pygame.event.get():
			match event:
				# check if the game should be closed
				case event if event.type == pygame.QUIT or \
					 (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
					sys.exit()

				# decrease the score of the players (event gets fired every second)
				case _ if event.type == DECREASE_SCORE_EVENT:
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
			bullet.draw(self.screen)

		# draw scoreboard
		self.scoreboard.draw_scoreboard(self.screen)

		pygame.display.flip()

	def _process_game_logic(self) -> None:
		"""private method to process game logic"""
		# update the times of every spaceship
		for spaceship in self.spaceships:
			spaceship.delta_time = spaceship.clock.tick(FRAMERATE)
			spaceship.time_elapsed_since_last_action += spaceship.delta_time
		
		# loop over every bullet and update its position
		for bullet in self.bullets:
			bullet.move()
			#If bullet get out of bound then delete it
			if bullet.x > self.screen.get_width() or \
			   bullet.x < 0 or \
			   bullet.y > self.screen.get_height() or \
			   bullet.y < 0:
				self.delete_bullet(bullet)

		# check for collisions of ships and bullets
		# self.scoreboard.decrease_score(ship.name, 100)
		# check if any ships are hit by any bullets
		for ship in self.spaceships:
			for bullet in self.bullets:
				if ship.hitbox.colliderect(bullet.hitbox):
					# check if bullet hit the shooter of the bullet itself
					if bullet.shooter == ship:
						continue
					# destroy bullet
					self.delete_bullet(bullet)
					# remove points from ship that got hit
					shooter_name = bullet.shooter.name
					shot_name = ship.name
					self.scoreboard.decrease_score(shot_name, self.POINTS_LOST_AFTER_GETTING_HIT)
					self.scoreboard.increase_score(shooter_name, self.POINTS_GAINED_AFTER_HITTING)

	def delete_bullet(self, bullet) -> None:
		self.bullets.remove(bullet)
		del bullet

	def spawn_spaceship(self, x: int, y: int, sprite: pygame.sprite.Sprite, \
						bullet_append_func: Callable[[Bullet], None], \
						screen: pygame.Surface,
				 		name: str) -> Spaceship:	
		spaceship = Spaceship(x, y, sprite, bullet_append_func, screen, name)
		self.spaceships.append(spaceship)
		self.scoreboard.attach(spaceship)
		return spaceship