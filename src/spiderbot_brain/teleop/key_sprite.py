"""Sprite class for drawing keys."""

import pygame


class KeySprite:
    """Sprite class for drawing keys."""

    def __init__(
            self,
            key1,
            key2,
            screen,
            x,
            y
            ):
        """Initialize the key sprite."""
        self.key1 = key1
        self.key2 = key2
        self.screen = screen
        self.x = x
        self.y = y

        self.key_background_up_color = (255, 255, 240)  # Ivory
        self.key_background_down_color = (128, 128, 128)  # Gray
        self.key_text_up_color = (0, 0, 0)  # Black
        self.key_text_down_color = (0, 0, 0)  # Black

        self.key_width = 50
        self.key_height = 50
        self.key_half_width = self.key_width / 2
        self.key_half_height = self.key_height / 2
        self.key_border_radius = 10

        self.font = pygame.font.Font(None, 48)
        self.key_text_offset_x = 25
        self.key_text_offset_y = 27

    def draw(self, is_key_down):
        """Draw the key sprite."""
        if is_key_down:
            pygame.draw.rect(
                self.screen,
                self.key_background_down_color,
                pygame.Rect(self.x, self.y, self.key_width, self.key_height),
                border_radius=self.key_border_radius
            )
            text_render = self.font.render(
                self.key1, True, self.key_text_down_color
            )
        else:
            pygame.draw.rect(
                self.screen,
                self.key_background_up_color,
                pygame.Rect(self.x, self.y, self.key_width, self.key_height),
                border_radius=self.key_border_radius
            )
            text_render = self.font.render(
                self.key1, True, self.key_text_up_color
            )
        text_rect = text_render.get_rect()
        text_rect.center = (
            self.x + self.key_text_offset_x, self.y + self.key_text_offset_y
        )
        self.screen.blit(text_render, text_rect)
