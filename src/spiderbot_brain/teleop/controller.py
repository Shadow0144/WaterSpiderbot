"""Creates a window and manages user input."""

import math
import time

import pygame

from .joystick_sprite import JoystickSprite
from .key_sprite import KeySprite


class Controller:
    """Create a window and manage user input."""

    def __init__(self, node):
        """Initialize the controller."""
        self.node = node

        self.running = True

        self.window_width = 275
        self.window_height = 130

        self.window_background_color = (119, 136, 153)  # Light Slate Gray

        target_fps = 60.0
        self.render_interval = 1.0 / target_fps
        self.last_render_time = time.time()

        self.spiderbot_body_x = 0.0
        self.spiderbot_body_y = 0.0
        self.spiderbot_yaw = 0.0

        self.keyboard_step_x = 0.1
        self.keyboard_step_y = 0.1
        self.keyboard_step_theta = math.pi / 8.0

        self.controller_step_x = 0.1
        self.controller_step_y = 0.1
        self.controller_step_theta = math.pi / 8.0

        self.controller_deadzone_x = 0.1
        self.controller_deadzone_y = 0.1

        self.pressed_keys = {
            'forward': False,
            'left': False,
            'backward': False,
            'right': False,
            'counterclockwise': False,
            'clockwise': False
        }

        pygame.init()
        pygame.joystick.init()

        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        pygame.display.set_caption('Spiderbot Controller')

        self.key_sprites = {
            'Q': KeySprite('Q', None, self.screen, 10, 10),
            'W': KeySprite('W', None, self.screen, 70, 10),
            'E': KeySprite('E', None, self.screen, 130, 10),
            'A': KeySprite('A', None, self.screen, 10, 70),
            'S': KeySprite('S', None, self.screen, 70, 70),
            'D': KeySprite('D', None, self.screen, 130, 70)
        }

        if pygame.joystick.get_count() > 0:
            self.left_joystick = pygame.joystick.Joystick(0)
        else:
            self.left_joystick = None
        if pygame.joystick.get_count() > 1:
            self.right_joystick = pygame.joystick.Joystick(1)
        else:
            self.right_joystick = None

        self.left_joystick_sprite = JoystickSprite(self.screen, 215, 35)
        self.right_joystick_sprite = JoystickSprite(self.screen, 240, 95)

        self.joystick_1_x = 0.0
        self.joystick_1_y = 0.0
        self.joystick_2_x = 0.0
        self.joystick_2_y = 0.0

    def destroy(self):
        """Clean up the pygame."""
        pygame.quit()

    def get_is_running(self):
        """Return if the window is still open and pygame is runnning."""
        return self.running

    def _poll_keyboard(self):
        """Get the keyboard state."""
        target_x = self.spiderbot_body_x
        target_y = self.spiderbot_body_y
        target_theta = self.spiderbot_yaw
        cos_yaw = math.cos(self.spiderbot_yaw)
        sin_yaw = math.sin(self.spiderbot_yaw)
        keys_pressed = pygame.key.get_pressed()
        if keys_pressed[pygame.K_w] or keys_pressed[pygame.K_UP]:
            target_x += self.keyboard_step_x * cos_yaw
            target_y += self.keyboard_step_y * sin_yaw
            self.pressed_keys['forward'] = True
        else:
            self.pressed_keys['forward'] = False
        if keys_pressed[pygame.K_a] or keys_pressed[pygame.K_LEFT]:
            target_x += self.keyboard_step_x * sin_yaw
            target_y += self.keyboard_step_y * cos_yaw
            self.pressed_keys['left'] = True
        else:
            self.pressed_keys['left'] = False
        if keys_pressed[pygame.K_s] or keys_pressed[pygame.K_DOWN]:
            target_x -= self.keyboard_step_x * cos_yaw
            target_y -= self.keyboard_step_y * sin_yaw
            self.pressed_keys['backward'] = True
        else:
            self.pressed_keys['backward'] = False
        if keys_pressed[pygame.K_d] or keys_pressed[pygame.K_RIGHT]:
            target_x -= self.keyboard_step_x * sin_yaw
            target_y -= self.keyboard_step_y * cos_yaw
            self.pressed_keys['right'] = True
        else:
            self.pressed_keys['right'] = False
        if keys_pressed[pygame.K_q] or keys_pressed[pygame.K_RCTRL]:
            target_theta += self.keyboard_step_theta
            self.pressed_keys['counterclockwise'] = True
        else:
            self.pressed_keys['counterclockwise'] = False
        if keys_pressed[pygame.K_e] or keys_pressed[pygame.K_KP0]:
            target_theta -= self.keyboard_step_theta
            self.pressed_keys['clockwise'] = True
        else:
            self.pressed_keys['clockwise'] = False
        target = [
            target_x,
            target_y,
            target_theta
        ]
        self.node.set_target(target)

    def _poll_controller(self):
        """Get the controller state."""
        if not self.left_joystick:
            return

        d_x = 0.0
        d_y = 0.0
        d_yaw = 0.0

        if self.left_joystick:
            if (
                math.abs(self.left_joystick.get_axis(0)) >
                self.controller_deadzone_x
            ):
                d_x = self.left_joystick.get_axis(0) * self.controller_step_x
            if (
                math.abs(self.left_joystick.get_axis(1)) >
                self.controller_deadzone_y
            ):
                d_y = self.left_joystick.get_axis(1) * self.controller_step_y
        if self.right_joystick:
            d_yaw = (
                self.right_joystick.get_axis(0) * self.controller_step_theta
            )

        self.joystick_1_x = d_x
        self.joystick_1_y = d_y
        self.joystick_2_x = d_yaw
        self.joystick_2_y = 0.0

        target_x = self.spiderbot_body_x
        target_y = self.spiderbot_body_y
        target_theta = self.spiderbot_yaw
        target_x += d_x * math.cos(self.spiderbot_yaw)
        target_y += d_y * math.sin(self.spiderbot_yaw)
        target_theta += d_yaw
        target = [
            target_x,
            target_y,
            target_theta
        ]
        self.node.set_target(target)

    def update_spiderbot_pose(self, pose):
        """Handle the updated Spiderbot pose."""
        # Position
        self.spiderbot_body_x = pose.position.x
        self.spiderbot_body_y = pose.position.y
        # Angle
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w
        self.spiderbot_yaw = (
            math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        )

        pygame.display.flip()

    def _draw(self):
        """Draw the window."""
        self.screen.fill(self.window_background_color)

        self.key_sprites['Q'].draw(self.pressed_keys['counterclockwise'])
        self.key_sprites['W'].draw(self.pressed_keys['forward'])
        self.key_sprites['E'].draw(self.pressed_keys['clockwise'])
        self.key_sprites['A'].draw(self.pressed_keys['left'])
        self.key_sprites['S'].draw(self.pressed_keys['backward'])
        self.key_sprites['D'].draw(self.pressed_keys['right'])

        self.left_joystick_sprite.draw(self.joystick_1_x, self.joystick_1_y)
        self.right_joystick_sprite.draw(self.joystick_2_x, self.joystick_2_y)

    def update(self):
        """Update the inputs and the window."""
        if not self.get_is_running():
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        current_timestamp = time.time()
        delta_time = current_timestamp - self.last_render_time
        if delta_time >= self.render_interval:
            self.last_render_time = current_timestamp

            pygame.event.pump()
            self._poll_keyboard()
            self._poll_controller()
            self._draw()
