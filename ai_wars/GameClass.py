import pygame
from pygame.math import Vector2

from .Utils import get_random_velocity, load_sprite, wrap_position, get_random_position
from .Spaceship import Spaceship
from .Bullet import Bullet
from .Action import EnumAction


class GameClass:

    def __init__(self):

        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.background = load_sprite("ai_wars/img/space.png", False)
        self.bullets = []
        self.spaceship = Spaceship(400, 300, 40, 40, load_sprite(
            "ai_wars/img/spaceship.png"), self.bullets.append, self.screen)
        #BORDER = pygame.Rect(800, 600)

    def main_loop(self):
        while True:
            self._handle_input()
            self._process_game_logic()
            self._draw()

    def _handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                quit()
            elif (
                self.spaceship
                and event.type == pygame.KEYDOWN
                and event.key == pygame.K_SPACE
            ):
                self.spaceship.shoot()

        is_key_pressed = pygame.key.get_pressed()

        if is_key_pressed[pygame.K_UP]:
            self.spaceship.move(EnumAction.FORWARD)

        if is_key_pressed[pygame.K_DOWN]:
            self.spaceship.move(EnumAction.BACKWARDS)

        if is_key_pressed[pygame.K_RIGHT]:
            self.spaceship.move(EnumAction.RIGHT)

        if is_key_pressed[pygame.K_LEFT]:
            self.spaceship.move(EnumAction.LEFT)

    def _draw(self):
        self.screen.blit(self.background, (0, 0))
        self.spaceship.draw(self.screen)

        for bullet in self.bullets:
            bullet.draw(self.screen)

        print("spacehshit drawn")
        pygame.display.flip()
        self.clock.tick(75)

    def _process_game_logic(self):
        for bullet in self.bullets:
            bullet.move()
