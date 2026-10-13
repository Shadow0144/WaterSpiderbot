"""A scene containing a simulated world and Spiderbot."""

import mujoco

from .scene_overlay import SceneOverlay


class Scene():
    """Scene class containing a Spiderbot."""

    def __init__(
            self,
            topic_prefix,
            simulated_spiderbot,
            train_parallel
    ):
        """Initialize the scene."""
        self.topic_prefix = topic_prefix
        self.simulated_spiderbot = simulated_spiderbot
        self.train_parallel = train_parallel

        self.model = simulated_spiderbot.model
        self.data = simulated_spiderbot.data

        self.scene_overlay = SceneOverlay(
            self.topic_prefix,
            self.train_parallel
        )

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(
            self.model,
            maxgeom=10_000
        )
        self.viewer_context = mujoco.MjrContext(
            self.model,
            mujoco.mjtFontScale.mjFONTSCALE_150
        )

        mujoco.mjv_defaultFreeCamera(
            self.model,
            self.cam
        )

        self.camera_following = True
        self.tracking_body_id = (
            self.model.body('cephalothorax').id
        )

        self.default_azimuth = 180
        self.default_elevation = -30
        self.default_distance = 3.0
        self.default_lookat = [0, 0, 0.25]

        self.reset_camera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = (self.tracking_body_id)

        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        self.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    def toggle_translucency(self):
        """Toggles if the models are translucent."""
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = (
            not self.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]
        )

    def toggle_camera_following(self):
        """Toggles if the camera is following the Spiderbot."""
        self.camera_following = not self.camera_following
        if self.camera_following:
            self.cam.type = (
                mujoco.mjtCamera.mjCAMERA_TRACKING
            )
            self.cam.trackbodyid = (
                self.tracking_body_id
            )
        else:
            self.cam.type = (
                mujoco.mjtCamera.mjCAMERA_FREE
            )

    def reset_camera(self):
        """Reset the camera's position and orientation."""
        self.cam.azimuth = self.default_azimuth
        self.cam.elevation = self.default_elevation
        self.cam.distance = self.default_distance
        self.cam.lookat[:] = self.default_lookat

    def set_training_configuration(
            self,
            num_targets_per_episode,
            num_episodes_per_candidate,
            num_candidates_per_generation,
            ):
        """Set the training configuration for the display."""
        self.scene_overlay.set_training_configuration(
            num_targets_per_episode,
            num_episodes_per_candidate,
            num_candidates_per_generation
        )

    def update_target_number(self, target_number):
        """Enable displaying the target number and update it."""
        self.scene_overlay.update_target_number(
            target_number
        )

    def update_training_status(self, training_status_msg):
        """Enable displaying the step reward and update it."""
        self.scene_overlay.update_training_status(
            training_status_msg
        )

    def _add_overlays(
            self,
            viewport
            ):
        """Add the overlay to the scene render."""
        self.scene_overlay.add_overlays(
            self.viewer_context,
            viewport
        )

    def update_scene(self, viewport):
        """Update and render the scene."""
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene
        )

        mujoco.mjr_render(
            viewport,
            self.scene,
            self.viewer_context
        )

        self._add_overlays(viewport)
