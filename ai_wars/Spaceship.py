import pygame
from pygame.math import Vector2

from .Action import EnumAction
from .Bullet import Bullet
from .Utils import load_sprite

UP = Vector2(0, -1)


class Spaceship():

    def __init__(self, x: float, y: float, height, width, sprite, bullet_append_func, surface):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.sprite = sprite
        self.bullet_append = bullet_append_func
        self.direction = Vector2(UP)
        self.surface = surface
        self.velocity = 2

    def draw(self, surface) -> None:
        player_rec = pygame.Rect(self.x, self.y, self.height, self.width)
        surface.blit(self.sprite, (self.x, self.y))

    def move(self, action):
        if action == EnumAction.FORWARD and self.y - self.velocity > 0:
            self.y -= self.velocity

        if action == EnumAction.BACKWARDS and self.y + self.velocity < \
                (self.surface.get_height() - self.height):
            self.y += self.velocity

        if action == EnumAction.RIGHT and self.x + self.velocity < \
                (self.surface.get_width() - self.width):
            self.x += self.velocity

        if action == EnumAction.LEFT and self.x - self.velocity > 0:
            self.x -= self.velocity

    def shoot(self):
        bullet_velocity = self.direction * 3
        bullet = Bullet(self.x, self.y, self.height, self.width,
                        load_sprite("ai_wars/img/bullet.png"), bullet_velocity)
        self.bullet_append(bullet)
        print("shots fired")
