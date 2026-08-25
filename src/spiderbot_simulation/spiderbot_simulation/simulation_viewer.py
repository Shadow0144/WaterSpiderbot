"""Viewer for rendering a Spiderbot simulation."""

import time

import glfw

import mujoco


class SimulationViewer():
    """Window and viewer for a Spiderbot simulation."""

    def __init__(self, model, data):
        """Initialize the window and viewer."""
        self.model = model
        self.data = data

        target_fps = 60.0
        self.render_interval = 1.0 / target_fps
        self.last_render_time = time.time()

        self.displaying_step_reward = False
        self.step_reward = 0.0
        self.displaying_episode_reward = False
        self.episode_reward = 0.0
        self.displaying_epoch_reward = False
        self.epoch_reward = 0.0

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

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10_000)
        self.viewer_context = mujoco.MjrContext(
            self.model,
            mujoco.mjtFontScale.mjFONTSCALE_150
        )

        mujoco.mjv_defaultFreeCamera(self.model, self.cam)

        self.cam.azimuth = 180
        self.cam.elevation = -20
        self.cam.distance = 2.0
        self.cam.lookat[:] = [0, 0, 0.25]

        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        self.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        self.mouse_button_left_pressed = False
        self.mouse_button_middle_pressed = False
        self.mouse_button_right_pressed = False
        self.mouse_last_x = 0
        self.mouse_last_y = 0

        glfw.set_cursor_pos_callback(self.window, self.mouse_move_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.window, self.mouse_scroll_callback)

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

        mujoco.mjv_moveCamera(
            self.model,
            int(action),
            mouse_dx / width,
            mouse_dy / height,
            self.cam
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
        mujoco.mjv_moveCamera(
            self.model,
            int(mujoco.mjtMouse.mjMOUSE_ZOOM),
            0,
            -0.05 * y_offset,
            self.cam
        )

    def _add_overlays(self, viewport):
        """Add the training overlays to the viewport."""
        labels = []
        values = []

        if self.displaying_step_reward:
            labels.append('Step reward:')
            values.append(f'{self.step_reward}')

        if self.displaying_episode_reward:
            labels.append('Episode reward:')
            values.append(f'{self.episode_reward}')

        if self.displaying_epoch_reward:
            labels.append('Epoch reward:')
            values.append(f'{self.epoch_reward}')

        if labels and values:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                '\n'.join(labels),
                '\n'.join(values),
                self.viewer_context
            )

    def update_step_reward(self, reward):
        """Enable displaying the step reward and update the reward value."""
        self.displaying_step_reward = True
        self.step_reward = reward

    def update_episode_reward(self, reward):
        """Enable displaying the episode reward and update the reward value."""
        self.displaying_episode_reward = True
        self.episode_reward = reward

    def update_epoch_reward(self, reward):
        """Enable displaying the epoch reward and update the reward value."""
        self.displaying_epoch_reward = True
        self.epoch_reward = reward

    def update(self, current_timestamp):
        """Update the render if enough time has elapsed."""
        if current_timestamp - self.last_render_time >= self.render_interval:
            self.last_render_time = current_timestamp
            width, height = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, width, height)

            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                None,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene
            )

            mujoco.mjr_render(viewport, self.scene, self.viewer_context)

            self._add_overlays(viewport)

            glfw.swap_buffers(self.window)
            glfw.poll_events()
