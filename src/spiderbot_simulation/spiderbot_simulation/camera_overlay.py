"""Renders what a camera views over the main scene."""

import mujoco


class CameraOverlay():
    """Overlay for rendering a camera's view."""

    def __init__(
            self,
            model,
            data,
            viewer_context,
            viewport,
            camera_name,
            camera_id
            ):
        """Initialize the overlay."""
        self.model = model
        self.data = data
        self.viewer_context = viewer_context
        self.viewport = viewport
        self.camera_name = camera_name
        self.camera_id = camera_id

        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.camera_id
        )
        
        self.opt = mujoco.MjvOption()

    def add_overlay(self):
        """Draw the camera overlay over the scene."""
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
            self.viewport,
            self.scene,
            self.viewer_context
        )

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            self.viewport,
            self.camera_name,
            None,
            self.viewer_context
        )
