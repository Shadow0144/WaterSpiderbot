"""Common class for a spider leg."""
import mujoco

import numpy as np

from .converters import matrix_to_rpy
from .converters import rpy_to_matrix


class SpiderLeg:
    """Common class for a spider leg."""

    def __init__(self, leg_id, segment_lengths, model, data):
        """Initialize a spider leg."""
        self.leg_id = leg_id
        self.segment_lengths = segment_lengths
        self.model = model
        self.data = data

        self.servo_coxa_actuator_id = self.model.actuator(
            'servo_' + self.leg_id + '_coxa_pitch').id
        self.servo_femur_actuator_id = self.model.actuator(
            'servo_' + self.leg_id + '_femur_pitch').id
        self.servo_tibia_actuator_id = self.model.actuator(
            'servo_' + self.leg_id + '_tibia_pitch').id

        coxa_joint_id = self.model.joint(
            f'{self.leg_id}_cephalothorax_coxa_joint').id
        femur_joint_id = self.model.joint(
            f'{self.leg_id}_coxa_femur_joint').id
        tibia_joint_id = self.model.joint(
            f'{self.leg_id}_femur_tibia_joint').id
        self.leg_joint_ids = [coxa_joint_id, femur_joint_id, tibia_joint_id]

        self.joint_limits = {}
        for joint_id in self.leg_joint_ids:
            self.joint_limits[joint_id] = self.model.jnt_range[joint_id]

        coxa_joint_dof = self.model.jnt_dofadr[coxa_joint_id]
        femur_joint_dof = self.model.jnt_dofadr[femur_joint_id]
        tibia_joint_dof = self.model.jnt_dofadr[tibia_joint_id]

        coxa_joint_qpos = self.model.jnt_qposadr[coxa_joint_id]
        femur_joint_qpos = self.model.jnt_qposadr[femur_joint_id]
        tibia_joint_qpos = self.model.jnt_qposadr[tibia_joint_id]

        self.leg_dof_adrs = [coxa_joint_dof,
                             femur_joint_dof,
                             tibia_joint_dof]
        self.leg_qpos_adrs = [coxa_joint_qpos,
                              femur_joint_qpos,
                              tibia_joint_qpos]

        self.coxa_body_id = self.model.body(f'{self.leg_id}_coxa').id

        self.leg_base_site_id = self.model.site(f'{self.leg_id}_leg_base').id
        self.claw_tip_site_id = self.model.site(f'{self.leg_id}_claw_tip').id

        target_mocap_body_id = self.model.body(f'{self.leg_id}_target').id
        self.target_geom_id = self.model.body_geomadr[target_mocap_body_id]
        self.target_mocap_id = self.model.body_mocapid[target_mocap_body_id]

        self.mocap_target_visible = True
        self.set_mocap_target_visible(False)  # Disable by default

        # Kinematics controls
        self.max_step = 0.05  # Max change in distance per control iteration
        self.max_dq_rad = 0.1  # Max change in angle per control iteration
        self.damping = 0.02  # Base damping factor
        self.filter_alpha = 0.3  # Low-pass filter for smoothing
        # Maintain a history to smooth next movements
        self.q_target = self.data.qpos[self.leg_qpos_adrs].copy()
        self.q_cmd_filtered = self.q_target.copy()

    def set_coxa_target(self, target_angle_rad):
        """Set the target angle for the coxa joint."""
        self.data.ctrl[self.servo_coxa_actuator_id] = target_angle_rad

    def set_femur_target(self, target_angle_rad):
        """Set the target angle for the femur joint."""
        self.data.ctrl[self.servo_femur_actuator_id] = target_angle_rad

    def set_tibia_target(self, target_angle_rad):
        """Set the target angle for the tibia joint."""
        self.data.ctrl[self.servo_tibia_actuator_id] = target_angle_rad

    def set_leg_targets(self, coxa_target_angle_rad,
                        femur_target_angle_rad,
                        tibia_target_angle_rad):
        """Set the target angles for the leg joints."""
        self.set_coxa_target(coxa_target_angle_rad)
        self.set_femur_target(femur_target_angle_rad)
        self.set_tibia_target(tibia_target_angle_rad)

    def get_qposes(self):
        """Return the qposes of all actuators."""
        return self.data.qpos[self.leg_qpos_adrs]

    def get_qvels(self):
        """Return the qvels of all actuators."""
        return self.data.qvel[self.leg_dof_adrs]

    def get_leg_base_xyz(self):
        """Return the position of the leg base."""
        return self.data.site_xpos[self.leg_base_site_id]

    def get_leg_base_rotation_matrix(self):
        """Return the rotation matrix the leg base."""
        mat = self.data.site_xmat[self.leg_base_site_id].reshape(3, 3)
        return mat

    def get_leg_base_rpy(self):
        """Return the roll, pitch, and yaw of the leg base."""
        mat = self.get_leg_base_rotation_matrix()
        return matrix_to_rpy(mat)

    def get_claw_xyz(self):
        """Return the position of the claw."""
        return self.data.site_xpos[self.claw_tip_site_id]

    def get_claw_rpy(self):
        """Return the roll, pitch, and yaw of the claw."""
        mat = self.data.site_xmat[self.claw_tip_site_id].reshape(3, 3)
        return matrix_to_rpy(mat)

    def set_target_qposes(self,
                          coxa_target_qpos,
                          femur_target_qpos,
                          tibia_target_qpos):
        """Set the target angles for the leg joints."""
        self.data.ctrl[self.servo_coxa_actuator_id] = coxa_target_qpos
        self.data.ctrl[self.servo_femur_actuator_id] = femur_target_qpos
        self.data.ctrl[self.servo_tibia_actuator_id] = tibia_target_qpos

    def set_mocap_target_visible(self, visible):
        """Toggle if the targets are visible."""
        if self.mocap_target_visible != visible:
            self.mocap_target_visible = visible
            alpha = 0.75 if visible else 0.0
            self.model.geom_rgba[self.target_geom_id, 3] = alpha

    def set_mocap_target(self, mocap_target_local):
        """Set the target angles for the leg joints."""
        leg_base_pos = self.get_leg_base_xyz()
        leg_base_rot = self.get_leg_base_rotation_matrix()
        mocap_target_world = (
            leg_base_pos + (leg_base_rot @ np.asarray(mocap_target_local))
        )
        self.data.mocap_pos[self.target_mocap_id] = mocap_target_world

    # target_xyz is a vector of the target x, y, and z position
    # target_rpy is desired the roll, pitch, and yaw of the end effector
    # relative is if the the target is relative to the leg base or not
    # max_ik_iterations is the maximum number of iterations to do while
    # solving the ik
    def move_claw_to_cartesian(self,
                               target_xyz,
                               target_rpy=None,
                               relative=True,
                               max_ik_iterations=5):
        """Move the tip of a leg to a point in Cartesian space."""
        # q_target = q_current + J^-1_p(x_target - x_current)
        # J is the Jacobian
        # q is the angles
        # x is the positions

        has_target_orientation = target_rpy is not None

        weight_pos = 1.0
        weight_rot = 0.2

        # Transform the targets into the correct space
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
        if has_target_orientation:
            r_target_local = rpy_to_matrix(target_rpy)
            if relative:
                r_target_world = r_leg_base @ r_target_local
            else:
                r_target_world = r_target_local

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

            current_xyz = self.data.site(
                self.claw_tip_site_id).xpos.copy()
            current_rpy = self.data.site(
                self.claw_tip_site_id).xmat.reshape(3, 3)

            # Get the Jacobian and then filter for the actuators we care about
            mujoco.mj_jacSite(self.model,
                              self.data,
                              jac_p, jac_r,
                              self.claw_tip_site_id)
            j_p = jac_p[:, self.leg_dof_adrs]
            j_r = jac_r[:, self.leg_dof_adrs]

            if relative:  # Move into relative space
                current_xyz = r_leg_base.T @ (current_xyz - leg_base_xyz)
                j_p = r_leg_base.T @ j_p
                j_r = r_leg_base.T @ j_r

            # Clamp the distance the leg attempts to travel
            delta_xyz = np.asarray(target_xyz) - current_xyz
            distance = np.linalg.norm(delta_xyz)
            if distance < 1e-4:
                break  # We've converged
            elif distance > self.max_step:
                # Avoid traveling too far in one step
                delta_xyz = delta_xyz * (self.max_step / distance)

            if has_target_orientation:
                if relative:
                    r_curr_eval = r_leg_base.T @ current_rpy
                    r_target_eval = r_target_local
                else:
                    r_curr_eval = current_rpy
                    r_target_eval = r_target_world

                r_err = r_target_eval @ r_curr_eval.T
                delta_r = 0.5 * np.array([
                    r_err[2, 1] - r_err[1, 2],
                    r_err[0, 2] - r_err[2, 0],
                    r_err[1, 0] - r_err[0, 1]
                ])
                distance_r = np.linalg.norm(delta_r)
                if distance_r > 0.1:
                    delta_r = delta_r * (0.1 / distance_r)
            else:
                delta_r = np.zeros(3)
                weight_rot = 0.0

            # Stack the vectors
            j_full = np.vstack([weight_pos * j_p,
                                weight_rot * j_r])
            delta_full = np.hstack([weight_pos * delta_xyz,
                                    weight_rot * delta_r])

            # Boost damping factor when determinant is too small
            jt_j = j_full.T @ j_full
            det_j = np.abs(np.linalg.det(jt_j))
            lambda_damping = self.damping + (0.05 if det_j < 1e-3 else 0.0)

            # Damped pseudo-inverse matrix
            j_inv = np.linalg.inv(jt_j + (lambda_damping**2) * np.eye(3))
            j_pinv = j_inv @ j_full.T

            # Compute step and update solution
            if not has_target_orientation:
                # Pull joints towards middle of joint limits
                mid_limits = np.array([
                    0.5 *
                    (self.joint_limits[joint_id][0] +
                     self.joint_limits[joint_id][1])
                    for joint_id in self.leg_joint_ids
                ])
                null_space_bias = 0.01 * (mid_limits - q_sol)
                null_projection = np.eye(3) - (j_pinv @ j_full)
                delta_q = (
                    j_pinv @ delta_full + (null_projection @ null_space_bias)
                )
            else:
                delta_q = j_pinv @ delta_full
            delta_q = np.clip(delta_q, -self.max_dq_rad, self.max_dq_rad)
            q_sol += delta_q

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

    def reset_leg(self):
        """Reset the leg to its initial state."""
        self.set_mocap_target_visible(False)
        self.q_target - self.data.qpos[self.leg_qpos_adrs].copy()
        self.q_cmd_filtered = self.q_target.copy()