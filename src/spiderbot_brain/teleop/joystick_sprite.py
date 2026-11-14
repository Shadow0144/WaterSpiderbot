"""Sprite class for drawing joysticks."""

import pygame


class JoystickSprite:
    """Sprite class for drawing a joystick."""

    def __init__(
            self,
            screen,
            x,
            y
            ):
        """Initialize the joystick sprite."""
        self.screen = screen
        self.x = x
        self.y = y

        self.offset_scaling = 0.1

        self.background_color = (0, 0, 0)  # Black
        self.joystick_color = (105, 105, 105)  # Dim Gray

        self.background_radius = 25
        self.joystick_radius = 20

    def draw(self, x_offset, y_offset):
        """Draw the joystick sprite."""
        pygame.draw.circle(
            self.screen,
            self.background_color,
            (self.x, self.y),
            self.background_radius
        )
        pygame.draw.circle(
            self.screen,
            self.joystick_color,
            (
                self.x + (x_offset * self.offset_scaling),
                self.y + (y_offset * self.offset_scaling)
            ),
            self.joystick_radius
        )
