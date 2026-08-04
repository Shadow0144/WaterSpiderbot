"""Resolves and keeps track of spider leg kinematics."""

import mujoco

import numpy as np


class SpiderLeg:
    """Resolves and keeps track of spider leg kinematics."""

    def __init__(self, leg_id, segment_length, model, data):
        """Create a spider leg."""
        self.leg_id = leg_id
        self.segment_length = segment_length
        self.model = model
        self.data = data

        self.damping = 0.01

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
        self.target_mocap_id = self.model.body_mocapid[target_mocap_body_id]

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

    def set_rest_targets(self, coxa_target_angle_rad,
                         femur_target_angle_rad,
                         tibia_target_angle_rad):
        """Set the angles for the leg joints to return to when resting."""
        self.rest_angles = {'coxa': coxa_target_angle_rad,
                            'femur': femur_target_angle_rad,
                            'tibia': tibia_target_angle_rad}

    def return_to_rest(self):
        """Move the leg joints to their rest positions."""
        self.set_leg_targets(self.rest_angles['coxa'],
                             self.rest_angles['femur'],
                             self.rest_angles['tibia'])

    def get_qposes(self):
        """Return the qposes of all actuators."""
        return self.data.qpos[self.leg_qpos_adrs]

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
