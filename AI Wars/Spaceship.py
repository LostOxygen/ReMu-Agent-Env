import pygame
from pygame.math import Vector2
from Action import Action
from Bullet import Bullet
from Utils import load_sprite

UP = Vector2(0, -1)


class Spaceship():

    def __init__(self, x: float, y: float, height, width, sprite, bullet_append_func, surface):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.sprite = sprite
        self.bullet_append = bullet_append_func  # self.bullet == self.Bullets.append
        self.direction = Vector2(UP)
        self.surface = surface
        self.velocity = 2

    def draw(self, surface) -> None:
        #player_rec = pygame.Rect(self.x, self.y, self.height, self.width)
        #surface.blit(self.sprite, (self.x, self.y))
        angle = self.direction.angle_to(UP)
        rotated_surface = pygame.transform.rotozoom(self.sprite, angle, 1.0)
        rotated_surface_size = Vector2(rotated_surface.get_size())
        blit_position = Vector2(self.x, self.y) - rotated_surface_size * 0.5
        surface.blit(rotated_surface, blit_position)

    def rotate(self, clockwise=True):
        sign = 1 if clockwise else -1
        angle = 3 * sign
        self.direction.rotate_ip(angle)

    def move(self, action):
        if action == Action.FORWARD and self.y - self.velocity > 0:
            self.y -= self.velocity

        if action == Action.BACKWARDS and self.y + self.velocity < \
                (self.surface.get_height() - self.height):
            self.y += self.velocity

        if action == Action.RIGHT and self.x + self.velocity < \
                (self.surface.get_width() - self.width):

            self.rotate(clockwise=True)

        if action == Action.LEFT and self.x - self.velocity > 0:
            self.rotate(clockwise=False)

    def shoot(self):
        bullet_velocity = self.direction * 3
        bullet = Bullet(self.x, self.y, self.height, self.width,
                        load_sprite("bullet"), bullet_velocity)
        self.bullet_append(bullet)
        print("shots fired")
