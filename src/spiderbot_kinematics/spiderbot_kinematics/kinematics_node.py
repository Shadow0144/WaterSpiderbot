"""Spiderbot kinematics node."""

import time

import mujoco

import rclpy
from rclpy.node import Node

from spiderbot_interfaces.msg import LegTargets
from spiderbot_interfaces.msg import SpiderbotTargetPose
from spiderbot_interfaces.srv import GetSpiderbotDescription

import spiderbot_utilities as utils

from .spider_leg import KinematicSpiderLeg


class SpiderbotKinematicsNode(Node):
    """Spiderbot locomotion."""

    def __init__(self):
        """Initialize and run a Spiderbot locomotor."""
        super().__init__('kinematics_node')

        self.spiderbot_description_client = self.create_client(
            GetSpiderbotDescription,
            'get_spiderbot_description')
        while not self.spiderbot_description_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info('Waiting on get_spec_xml service')
        spiderbot_description = self.request_spiderbot_description()

        self.leg_names = spiderbot_description.leg_names
        self.segment_lengths = dict(zip(self.leg_names,
                                        spiderbot_description.segment_lengths))
        self.spec = mujoco.MjSpec.from_string(
            spiderbot_description.spec_xml
        )
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        body_joint_id = self.model.joint('cephalothorax_joint').id
        self.body_joint_qpos_adr = self.model.jnt_qposadr[body_joint_id]
        self.legs = {}
        for leg_name in self.leg_names:
            self.legs[leg_name] = KinematicSpiderLeg(
                leg_name,
                self.segment_lengths[leg_name],
                self.model,
                self.data)

        self.set_leg_targets_subscription = self.create_subscription(
            LegTargets,
            'set_leg_targets',
            self.set_leg_targets_callback,
            10
        )
        self.set_leg_targets_subscription

        self.spiderbot_target_pose_publisher = self.create_publisher(
            SpiderbotTargetPose, 'spiderbot_target_pose', 10)

    def request_spiderbot_description(self):
        """Get the spec xml from the description."""
        request = GetSpiderbotDescription.Request()
        future = self.spiderbot_description_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def set_leg_targets_callback(self, msg):
        """Determine the poses required to best reach the targets."""
        leg_target_values = msg.leg_targets
        leg_targets = dict(zip(self.leg_names, leg_target_values))
        for leg_name in self.leg_names:
            # Move the claw to the target point
            self.legs[leg_name].move_claw_to_cartesian(
                utils.convert_vector3_to_list(leg_targets[leg_name]))
        mujoco.mj_step(self.model, self.data)

        pose_msg = utils.construct_target_pose_msg_from_legs(
            time.time(),
            self.leg_names,
            self.legs
        )
        self.spiderbot_target_pose_publisher.publish(pose_msg)
