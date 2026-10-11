"""A simulated Spiderbot for the simuation."""

import math

import mujoco

import numpy as np

import spiderbot_utilities as utils
from spiderbot_utilities import SpiderLeg


class SimulatedSpiderbot():
    """Simulated Spiderbot."""

    def __init__(self, topic_prefix, description):
        """Initialize the Spiderbot."""
        self.topic_prefix = topic_prefix
        self.description = description

        spec = mujoco.MjSpec.from_string(
            self.description.spec_xml
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.body = self.data.body('cephalothorax')

        # Create the legs and leg descriptions
        self.leg_descriptions = self.description.leg_descriptions
        self.leg_names = [
            leg_description.leg_name
            for leg_description in self.leg_descriptions
        ]
        self.segment_lengths_per_leg = {}
        for i, leg_name in enumerate(self.leg_names):
            self.segment_lengths_per_leg[leg_name] = (
                self.leg_descriptions[i].segment_lengths
            )
        self.legs = {}
        for leg_name in self.leg_names:
            self.legs[leg_name] = SpiderLeg(
                leg_name,
                self.segment_lengths_per_leg[leg_name],
                self.model,
                self.data
            )

        self.target_visible = False
        self.target_position = None
        self.target_quaternion = None
        self.target = self.data.body('target')
        self.target_z = self.model.body('target').pos[2]
        target_body_id = self.model.body('target').id
        self.target_mocap_id = self.model.body_mocapid[
            target_body_id
        ]
        self.target_forward_geom_id = self.model.geom(
            'target_forward_geom'
        ).id
        self.target_backward_geom_id = self.model.geom(
            'target_backward_geom'
        ).id

        self.pose_arrow_position = None
        self.pose_arrow_quaternion = None
        self.pose_arrow = self.data.body('pose_arrow')
        self.pose_arrow_z = self.model.body('pose_arrow').pos[2]
        pose_arrow_body_id = self.model.body('pose_arrow').id
        self.pose_arrow_mocap_id = self.model.body_mocapid[
            pose_arrow_body_id
        ]
        self.pose_arrow_forward_geom_id = self.model.geom(
            'pose_arrow_forward_geom'
        ).id
        self.pose_arrow_backward_geom_id = self.model.geom(
            'pose_arrow_backward_geom'
        ).id

    def reset(self):
        """Reset the Spiderbot to its starting state."""
        mujoco.mj_resetData(self.model, self.data)
        # Move the target back to where it should be if necessary
        if self.target_visible:
            self.data.mocap_pos[self.target_mocap_id] = (
                self.target_position
            )
            self.data.mocap_quat[self.target_mocap_id] = (
                self.target_quaternion
            )
        mujoco.mj_forward(self.model, self.data)
        for leg_name in self.leg_names:
            self.legs[leg_name].reset_leg()

    def set_leg_targets(self, targets_msg):
        """Move the mocaps to the targets."""
        leg_target_values = targets_msg.leg_targets
        if len(leg_target_values) != len(self.leg_names):
            return  # Break early
        leg_targets = dict(zip(self.leg_names, leg_target_values))
        for leg_name in self.leg_names:
            # Move the mocap to the target point
            self.legs[leg_name].set_mocap_target_visible(True)
            self.legs[leg_name].set_mocap_target(
                utils.convert_vector3_to_list(leg_targets[leg_name])
            )

    def set_target_pose(self, target_pose_msg):
        """Set the target pose."""
        leg_poses_values = target_pose_msg.leg_poses
        if len(leg_poses_values) != len(self.leg_names):
            return  # Break early
        leg_poses = dict(zip(self.leg_names, leg_poses_values))
        if target_pose_msg.poses_normalized:
            for leg_name in self.leg_names:
                leg_pose = leg_poses[leg_name]
                self.legs[leg_name].set_leg_actions(
                    leg_pose.coxa_qpos,
                    leg_pose.femur_qpos,
                    leg_pose.tibia_qpos
                )
        else:
            for leg_name in self.leg_names:
                leg_pose = leg_poses[leg_name]
                self.legs[leg_name].set_leg_target_angles(
                    leg_pose.coxa_qpos,
                    leg_pose.femur_qpos,
                    leg_pose.tibia_qpos
                )

    def set_target(self, target_msg):
        """Set the target for locomotion."""
        self.target_visible = True
        alpha = 0.75
        self.target_position = [
            target_msg.target_x,
            target_msg.target_y,
            self.target_z
        ]
        half_theta = target_msg.target_theta / 2.0
        self.target_quaternion = [
            np.cos(half_theta), 0.0, 0.0, np.sin(half_theta)
        ]
        self.model.geom_rgba[self.target_forward_geom_id, 3] = alpha
        self.model.geom_rgba[self.target_backward_geom_id, 3] = alpha
        self.model.geom_rgba[self.pose_arrow_forward_geom_id, 3] = alpha
        self.model.geom_rgba[self.pose_arrow_backward_geom_id, 3] = alpha
        self.data.mocap_pos[self.target_mocap_id] = (
                    self.target_position
                )
        self.data.mocap_quat[self.target_mocap_id] = (
            self.target_quaternion
        )

    def create_pose_message(self, last_timestamp):
        """Get the pose as a message."""
        spiderbot_pose_msg = utils.construct_pose_msg(
            last_timestamp,
            self.body,
            self.leg_names,
            self.legs
        )
        return spiderbot_pose_msg

    def _update_pose_arrow(self):
        """Move the pose arrows to match the pose."""
        self.pose_arrow_position = [
            self.body.xpos[0],
            self.body.xpos[1],
            self.pose_arrow_z
        ]
        self.data.mocap_pos[self.pose_arrow_mocap_id] = (
                            self.pose_arrow_position
        )

        qx = self.body.xquat[1]
        qy = self.body.xquat[2]
        qz = self.body.xquat[3]
        qw = self.body.xquat[0]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        half_yaw = yaw * 0.5
        yaw_quaternion = np.array([
            np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)
        ])
        self.pose_arrow_quaternion = yaw_quaternion
        self.data.mocap_quat[self.pose_arrow_mocap_id] = (
            self.pose_arrow_quaternion
        )

    def step(self):
        """Step the simulation."""
        mujoco.mj_step(self.model, self.data)
        self._update_pose_arrow()
