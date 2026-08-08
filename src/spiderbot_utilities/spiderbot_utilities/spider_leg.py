"""Common class for a spider leg."""

import numpy as np

from .converters import matrix_to_rpy


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

    def get_leg_base_rpy(self):
        """Return the roll, pitch, and yaw of the leg base."""
        mat = self.data.site_xmat[self.leg_base_site_id].reshape(3, 3)
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
        leg_base_rot = self.get_leg_base_rpy()
        mocap_target_world = (
            leg_base_pos + (leg_base_rot @ np.asarray(mocap_target_local))
        )
        self.data.mocap_pos[self.target_mocap_id] = mocap_target_world

    def reset_leg(self):
        """Reset the leg to its initial state."""
        self.set_mocap_target_visible(False)
