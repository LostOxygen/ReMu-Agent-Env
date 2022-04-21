import pygame
from pygame.math import Vector2
from Utils import load_sprite


class Bullet():
    def __init__(self, x, y, height, width, sprite, velocity: Vector2):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.sprite = sprite
        self.velocity = velocity

    def move(self):
        self.x = self.x + self.velocity.x
        self.y = self.y + self.velocity.y

    def draw(self, surface):
        bullet_position = pygame.Rect(self.x, self.y, self.height, self.width)
        surface.blit(self.sprite, (self.x, self.y))
