"""Create a leg for quickly referencing joints."""

import numpy as np


class SpiderLeg:
    """A single Spiderbot leg with 3 joints."""

    def __init__(self, leg_id, segment_length, model, data):
        """Initialize the spider leg."""
        self.leg_id = leg_id
        self.segment_length = segment_length
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

    def get_qposes(self):
        """Return the qposes of all actuators."""
        return self.data.qpos[self.leg_qpos_adrs]

    def set_target_qposes(self,
                          coxa_target_qpos,
                          femur_target_qpos,
                          tibia_target_qpos):
        """Set the target angles for the leg joints."""
        self.data.ctrl[self.servo_coxa_actuator_id] = coxa_target_qpos
        self.data.ctrl[self.servo_femur_actuator_id] = femur_target_qpos
        self.data.ctrl[self.servo_tibia_actuator_id] = tibia_target_qpos

    def set_mocap_target(self, mocap_target_local):
        """Set the target angles for the leg joints."""
        leg_base_pos = self.data.site_xpos[self.leg_base_site_id]
        leg_base_rot = self.data.site_xmat[self.leg_base_site_id].reshape(3, 3)
        mocap_target_world = (
            leg_base_pos + (leg_base_rot @ np.asarray(mocap_target_local))
        )
        self.data.mocap_pos[self.target_mocap_id] = mocap_target_world
