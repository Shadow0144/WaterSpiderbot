"""Resolves and keeps track of spider leg kinematics."""

import mujoco

import numpy as np

from spiderbot_utilities import SpiderLeg


class KinematicSpiderLeg(SpiderLeg):
    """Resolves and keeps track of spider leg kinematics."""

    def __init__(self, leg_id, segment_length, model, data):
        """Initialize a spider leg."""
        super().__init__(leg_id, segment_length, model, data)

    # target_xyz is a vector of the target x, y, and z position
    # target_rpy is desired the roll, pitch, and yaw of the end effector
    def move_claw_to_cartesian(self, target_xyz, target_rpy=[], relative=True):
        """Move the tip of a leg to a point in Cartesian space."""
        # q_target = q_current + J^-1_p(x_target - x_current)
        # J is the Jacobian
        # q is the angles
        # x is the positions

        current_xyz = self.data.site(self.claw_tip_site_id).xpos

        # Get the Jacobian and then filter for the actuators we care about
        jac_p = np.zeros((3, self.model.nv))
        jac_r = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data,
                          jac_p, jac_r, self.claw_tip_site_id)
        j_leg = jac_p[:, self.leg_dof_adrs]

        if relative:
            r_leg_base = (
                self.data.site(self.leg_base_site_id).xmat.reshape(3, 3)
            )
            leg_base_xyz = self.data.site(self.leg_base_site_id).xpos
            current_xyz = r_leg_base.T @ (current_xyz - leg_base_xyz)
            j_leg = r_leg_base.T @ j_leg
            self.data.mocap_pos[self.target_mocap_id] = (
                leg_base_xyz + (r_leg_base @ target_xyz)
            )
        else:
            self.data.mocap_pos[self.target_mocap_id] = target_xyz

        # Damped pseudo-inverse matrix
        j_jt = j_leg @ j_leg.T + (self.damping**2) * np.eye(3)
        j_pinv = j_leg.T @ np.linalg.inv(j_jt)

        # Change in angles
        dxyz = (target_xyz - current_xyz).T
        d_q = j_pinv @ dxyz
        q_target = self.data.qpos[self.leg_qpos_adrs] + d_q

        coxa_target = q_target[0]
        femur_target = q_target[1]
        tibia_target = q_target[2]

        self.set_leg_targets(coxa_target, femur_target, tibia_target)
