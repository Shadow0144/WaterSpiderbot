"""Resolves and keeps track of spider leg kinematics."""

import mujoco

import numpy as np

from spiderbot_utilities import SpiderLeg


class KinematicSpiderLeg(SpiderLeg):
    """Resolves and keeps track of spider leg kinematics."""

    def __init__(self, leg_id, segment_length, model, data):
        """Initialize a spider leg."""
        super().__init__(leg_id, segment_length, model, data)
        self.max_step = 0.05  # Max change in distance per control iteration
        self.max_dq_rad = 0.1  # Max change in angle per control iteration
        self.damping = 0.02  # Base damping factor
        self.filter_alpha = 0.3  # Low-pass filter for smoothing

        # Maintain a history to smooth next movements
        self.q_target = self.data.qpos[self.leg_qpos_adrs].copy()
        self.q_cmd_filtered = self.q_target.copy()

    # target_xyz is a vector of the target x, y, and z position
    # target_rpy is desired the roll, pitch, and yaw of the end effector
    # relative is if the the target is relative to the leg base or not
    # max_ik_iterations is the maximum number of iterations to do while
    # solving the ik
    def move_claw_to_cartesian(self,
                               target_xyz,
                               target_rpy=[],
                               relative=True,
                               max_ik_iterations=5):
        """Move the tip of a leg to a point in Cartesian space."""
        # q_target = q_current + J^-1_p(x_target - x_current)
        # J is the Jacobian
        # q is the angles
        # x is the positions

        if relative:
            r_leg_base = (
                self.data.site(self.leg_base_site_id).xmat.reshape(3, 3)
            )
            leg_base_xyz = self.data.site(self.leg_base_site_id).xpos
            self.data.mocap_pos[self.target_mocap_id] = (
                leg_base_xyz + (r_leg_base @ target_xyz)
            )
        else:
            self.data.mocap_pos[self.target_mocap_id] = target_xyz

        q_original = self.data.qpos[self.leg_qpos_adrs].copy()
        q_sol = q_original.copy()

        jac_p = np.zeros((3, self.model.nv))
        jac_r = np.zeros((3, self.model.nv))

        # Loop through solving the ik, this will help ensure the legs are able
        # to resist gravity without jittering around
        for _ in range(max_ik_iterations):
            # Set the virtual state and run forward kinematics
            self.data.qpos[self.leg_qpos_adrs] = q_sol
            mujoco.mj_kinematics(self.model, self.data)

            current_xyz = self.data.site(self.claw_tip_site_id).xpos.copy()

            # Get the Jacobian and then filter for the actuators we care about
            mujoco.mj_jacSite(self.model,
                              self.data,
                              jac_p, jac_r,
                              self.claw_tip_site_id)
            j_leg = jac_p[:, self.leg_dof_adrs]

            if relative:  # Move into relative space
                current_xyz = r_leg_base.T @ (current_xyz - leg_base_xyz)
                j_leg = r_leg_base.T @ j_leg

            # Clamp the distance the leg attempts to travel
            dxyz = np.asarray(target_xyz) - current_xyz
            distance = np.linalg.norm(dxyz)
            if distance < 1e-4:
                break  # We've converged
            elif distance > self.max_step:
                # Avoid traveling too far in one step
                dxyz = dxyz * (self.max_step / distance)

            # Boost damping factor when determinant is too small
            det_j = np.abs(np.linalg.det(j_leg))
            lambda_damping = self.damping + (0.05 if det_j < 1e-3 else 0.0)

            # Damped pseudo-inverse matrix
            j_jt = j_leg @ j_leg.T + (lambda_damping**2) * np.eye(3)
            j_pinv = j_leg.T @ np.linalg.inv(j_jt)

            # Pull joints towards middle of joint limits
            mid_limits = np.array([
                (self.joint_limits[joint_id][0] +
                 (self.joint_limits[joint_id][1] / 2.0))
                for joint_id in self.leg_joint_ids
            ])
            null_space_bias = 0.01 * (mid_limits - q_sol)
            null_projection = np.eye(3) - (j_pinv @ j_leg)

            # Compute step and update solution
            d_q = j_pinv @ dxyz + (null_projection @ null_space_bias)
            d_q = np.clip(d_q, -self.max_dq_rad, self.max_dq_rad)
            q_sol += d_q

            # Clamp solution to joint limits
            for i, joint_id in enumerate(self.leg_joint_ids):
                q_sol[i] = np.clip(q_sol[i],
                                   self.joint_limits[joint_id][0],
                                   self.joint_limits[joint_id][1])

        # Restore the legs after the calculations
        self.data.qpos[self.leg_qpos_adrs] = q_original
        mujoco.mj_kinematics(self.model, self.data)

        self.q_target = q_sol.copy()

        # Apply low-pass filter to remove actuator steps
        self.q_cmd_filtered = (self.filter_alpha * self.q_target) + (
            (1.0 - self.filter_alpha) * self.q_cmd_filtered
        )

        self.set_leg_targets(
            self.q_cmd_filtered[0],
            self.q_cmd_filtered[1],
            self.q_cmd_filtered[2])
