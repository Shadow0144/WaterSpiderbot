"""Provide a teleop interface for a Spiderbot."""

import math
import time

import glfw

import pygame

from rclpy.node import Node

from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.msg import Target
from spiderbot_interfaces.srv import GetTrainingConfiguration


class TeleopNode(Node):
    """A teleop node for a Spiderbot."""

    def __init__(self):
        """Initialize the teleop node."""
        super().__init__('teleop_node')

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
            glfw.KEY_W: False,
            glfw.KEY_UP: False,
            glfw.KEY_A: False,
            glfw.KEY_LEFT: False,
            glfw.KEY_S: False,
            glfw.KEY_DOWN: False,
            glfw.KEY_D: False,
            glfw.KEY_RIGHT: False,
            glfw.KEY_Q: False,
            glfw.KEY_RIGHT_CONTROL: False,
            glfw.KEY_E: False,
            glfw.KEY_KP_0: False,
        }

        glfw.init()
        self.window = glfw.create_window(
            300,
            150,
            'Spiderbot Teleop',
            None,
            None
        )
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        glfw.set_key_callback(self.window, self.key_callback)

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() > 0:
            self.left_joystick = pygame.joystick.Joystick(0)
        else:
            self.left_joystick = None
        if pygame.joystick.get_count() > 1:
            self.right_joystick = pygame.joystick.Joystick(1)
        else:
            self.right_joystick = None

        self.target_publisher = self.create_publisher(
            Target,
            'target',
            10
        )

        self.spiderbot_pose_subscription = self.create_subscription(
            SpiderbotPose,
            'spiderbot_pose',
            self.spiderbot_pose_callback,
            10
        )

        self.get_training_configuration_service = self.create_service(
            GetTrainingConfiguration,
            'get_training_configuration',
            self.get_training_configuration_callback
        )

    def destroy_node(self):
        """Destroy the window and finish destroying the node."""
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
        return super().destroy_node()

    def is_running(self):
        """Return if the node is running."""
        return not glfw.window_should_close(self.window)

    def get_training_configuration_callback(self, request, response):
        """Provide the training configuration."""
        response.training_mode_enabled = False
        response.num_targets_per_episode = 0
        response.num_episodes_per_candidate = 0
        response.num_candidates_per_generation = 0
        return response

    def _set_target(self, target):
        """Publish a new target."""
        msg = Target()
        msg.target_x = target[0]
        msg.target_y = target[1]
        msg.target_theta = target[2]
        self.target_publisher.publish(msg)
        self.get_logger().info(
            f'Setting target '
            f'{[f"{item:.6f}" for item in target]}',
            throttle_duration_sec=3.0
        )

    def key_callback(self, window, key, scancode, action, mods):
        """Handle key events in the window."""
        if key in self.pressed_keys.keys():
            self.pressed_keys[key] = (
                action == glfw.PRESS or action == glfw.REPEAT
            )
        self._handle_keys()

    def _handle_keys(self):
        """Handle the keys."""
        target_x = self.spiderbot_body_x
        target_y = self.spiderbot_body_y
        target_theta = self.spiderbot_yaw
        cos_yaw = math.cos(self.spiderbot_yaw)
        sin_yaw = math.sin(self.spiderbot_yaw)
        for key, value in self.pressed_keys.items():
            if value:
                if key == glfw.KEY_W or key == glfw.KEY_UP:
                    target_x += self.keyboard_step_x * cos_yaw
                    target_y += self.keyboard_step_y * sin_yaw
                if key == glfw.KEY_A or key == glfw.KEY_LEFT:
                    target_x += self.keyboard_step_x * sin_yaw
                    target_y += self.keyboard_step_y * cos_yaw
                if key == glfw.KEY_S or key == glfw.KEY_DOWN:
                    target_x -= self.keyboard_step_x * cos_yaw
                    target_y -= self.keyboard_step_y * sin_yaw
                if key == glfw.KEY_D or key == glfw.KEY_RIGHT:
                    target_x -= self.keyboard_step_x * sin_yaw
                    target_y -= self.keyboard_step_y * cos_yaw
                if key == glfw.KEY_Q or key == glfw.KEY_RIGHT_CONTROL:
                    target_theta += self.keyboard_step_theta
                if key == glfw.KEY_E or key == glfw.KEY_KP_0:
                    target_theta -= self.keyboard_step_theta
        target = [
            target_x,
            target_y,
            target_theta
        ]
        self._set_target(target)

    def _poll_controller(self):
        """Get the controller state."""
        if not self.left_joystick:
            return

        self.get_logger().info('Polling')

        pygame.event.pump()

        d_x = 0
        d_y = 0
        d_yaw = 0

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
        self._set_target(target)

    def spiderbot_pose_callback(self, msg):
        """Handle the updated Spiderbot pose."""
        # Update the stored position of the Spiderbot
        pose = msg.body_odometry.pose.pose
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

    def update(self):
        """Update the node and viewer."""
        if not self.is_running():
            return

        current_timestamp = time.time()
        if current_timestamp - self.last_render_time >= self.render_interval:
            self.last_render_time = current_timestamp

            self._poll_controller()

            glfw.swap_buffers(self.window)
            glfw.poll_events()
