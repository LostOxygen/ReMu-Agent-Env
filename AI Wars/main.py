import pygame
from Utils import get_random_velocity, load_sprite, wrap_position, get_random_position
from pygame.math import Vector2
from Spaceship import Spaceship
from Bullet import Bullet
from Action import Action


class AI_Wars:

    def __init__(self):

        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.background = load_sprite("space", False)
        self.bullets = []
        self.spaceship = Spaceship(400, 300, 40, 40, load_sprite(
            "spaceship"), self.bullets.append, self.screen)
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
            self.spaceship.move(Action.FORWARD)

        if is_key_pressed[pygame.K_DOWN]:
            self.spaceship.move(Action.BACKWARDS)

        if is_key_pressed[pygame.K_RIGHT]:
            self.spaceship.move(Action.RIGHT)

        if is_key_pressed[pygame.K_LEFT]:
            self.spaceship.move(Action.LEFT)

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


if __name__ == "__main__":
    AI_WARS = AI_Wars()
    AI_WARS.main_loop()
