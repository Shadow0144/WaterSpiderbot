"""Viewer for rendering a Spiderbot simulation."""

import time

import glfw

import mujoco

from .scene import Scene


class SimulationViewer():
    """Window and viewer for a Spiderbot simulation."""

    def __init__(
            self,
            topic_prefixes,
            simulated_spiderbots,
            train_parallel
            ):
        """Initialize the window and viewer."""
        self.topic_prefixes = topic_prefixes
        self.simulated_spiderbots = simulated_spiderbots
        self.train_parallel = train_parallel

        if not self.topic_prefixes or not self.simulated_spiderbots:
            raise RuntimeError(
                'There must be at least one topic and spiderbot'
            )

        target_fps = 60.0
        self.render_interval = 1.0 / target_fps
        self.last_render_time = time.time()

        self.current_topic_index = 0

        glfw.init()
        self.window = glfw.create_window(
            1200,
            900,
            'Spiderbot Simulation',
            None,
            None
        )
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        self.num_scenes = len(self.topic_prefixes)
        self.scenes = {}
        for topic_prefix in self.topic_prefixes:
            if topic_prefix in self.simulated_spiderbots.keys():
                self.scenes[topic_prefix] = Scene(
                    topic_prefix,
                    self.simulated_spiderbots[topic_prefix],
                    self.train_parallel
                )

        self.mouse_button_left_pressed = False
        self.mouse_button_middle_pressed = False
        self.mouse_button_right_pressed = False
        self.mouse_last_x = 0
        self.mouse_last_y = 0

        glfw.set_cursor_pos_callback(self.window, self.mouse_move_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.window, self.mouse_scroll_callback)
        glfw.set_key_callback(self.window, self.key_callback)

    def destroy(self):
        """Destroy the window and terminate glfw."""
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()

    def is_running(self):
        """Return True if the window is not ready to close."""
        return not glfw.window_should_close(self.window)

    def mouse_move_callback(self, window, xpos, ypos):
        """Handle mouse move actions in the window."""
        mouse_dx = xpos - self.mouse_last_x
        mouse_dy = ypos - self.mouse_last_y
        self.mouse_last_x = xpos
        self.mouse_last_y = ypos

        if not (
            self.mouse_button_left_pressed or
            self.mouse_button_middle_pressed or
            self.mouse_button_right_pressed
        ):
            return

        width, height = glfw.get_window_size(window)
        if width <= 0 or height <= 0:
            return

        mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                     glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

        if self.mouse_button_right_pressed:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if mod_shift else
                mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif self.mouse_button_left_pressed:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if mod_shift else
                mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        elif self.mouse_button_middle_pressed:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        current_scene = self._get_current_scene()
        mujoco.mjv_moveCamera(
            current_scene.model,
            int(action),
            mouse_dx / width,
            mouse_dy / height,
            current_scene.cam
        )

    def mouse_button_callback(self, window, button, action, mods):
        """Handle mouse button actions in the window."""
        self.mouse_button_left_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)
            == glfw.PRESS
        )
        self.mouse_button_middle_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE)
            == glfw.PRESS
        )
        self.mouse_button_right_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT)
            == glfw.PRESS
        )
        self.mouse_last_x, self.mouse_last_y = glfw.get_cursor_pos(window)

    def mouse_scroll_callback(self, window, x_offset, y_offset):
        """Handle mouse scroll in the window."""
        current_scene = self._get_current_scene()
        mujoco.mjv_moveCamera(
            current_scene.model,
            int(mujoco.mjtMouse.mjMOUSE_ZOOM),
            0,
            -0.05 * y_offset,
            current_scene.cam
        )

    def key_callback(self, window, key, scancode, action, mods):
        """Handle key events in the window."""
        current_scene = self._get_current_scene()
        if action == glfw.PRESS:
            if key == glfw.KEY_LEFT:
                # Decrease the scene index
                self.current_topic_index = (
                    (self.current_topic_index - 1) % self.num_scenes
                )  # Python will preserve the sign of the divisor
            if key == glfw.KEY_RIGHT:
                # Increase the scene index
                self.current_topic_index = (
                    (self.current_topic_index + 1) % self.num_scenes
                )
            elif key == glfw.KEY_T:
                # Toggle translucency
                current_scene.toggle_translucency()
            elif key == glfw.KEY_J:
                # Toggle joint visibility
                current_scene.toggle_joints_visible()
            elif key == glfw.KEY_V:
                # Toggle frame visibility
                current_scene.toggle_frames_visible()
            elif key == glfw.KEY_F:
                # Follow camera tracking/following
                current_scene.toggle_camera_following()
            elif key == glfw.KEY_R:
                # Reset the camera
                current_scene.reset_camera()

    def _get_current_scene(self):
        """Shorthand for getting the current scene."""
        return self.scenes[self.topic_prefixes[self.current_topic_index]]

    def set_training_configuration(
            self,
            num_targets_per_episode,
            num_episodes_per_candidate,
            num_candidates_per_generation,
            ):
        """Set the training configuration for the display."""
        for scene in self.scenes.values():
            scene.set_training_configuration(
                num_targets_per_episode,
                num_episodes_per_candidate,
                num_candidates_per_generation
            )

    def update_target_number(self, topic_prefix, target_number):
        """Enable displaying the target number and update it."""
        if topic_prefix in self.scenes.keys():
            self.scenes[topic_prefix].update_target_number(target_number)

    def update_training_status(self, topic_prefix, training_status_msg):
        """Enable displaying the step reward and update it."""
        if topic_prefix in self.scenes.keys():
            self.scenes[topic_prefix].update_training_status(
                training_status_msg
            )

    def update(self, current_timestamp):
        """Update the render if enough time has elapsed."""
        if (current_timestamp - self.last_render_time) >= self.render_interval:
            self.last_render_time = current_timestamp
            width, height = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, width, height)

            self._get_current_scene().update_scene(viewport)

            glfw.swap_buffers(self.window)
            glfw.poll_events()
