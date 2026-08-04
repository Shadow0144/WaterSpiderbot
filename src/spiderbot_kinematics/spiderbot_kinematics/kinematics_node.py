"""Spiderbot kinematics node."""

import time

import mujoco

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState

from spiderbot_interfaces.msg import LegPose
from spiderbot_interfaces.msg import LegTargets
from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.srv import GetSpiderbotDescription

from .spider_leg import SpiderLeg


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
            self.legs[leg_name] = SpiderLeg(leg_name,
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
            SpiderbotPose, 'spiderbot_target_pose', 10)

    def request_spiderbot_description(self):
        """Get the spec xml from the description."""
        request = GetSpiderbotDescription.Request()
        future = self.spiderbot_description_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def create_joint_state(self, name, qpose):
        """Construct a JointStateObject."""
        joint_state = JointState()
        joint_state.name = name
        joint_state.position = qpose
        return joint_state

    def construct_pose_msg(self):
        """Construct a pose message."""
        msg = SpiderbotPose()
        msg.timestamp = time.time()
        msg.body_joint_state = self.create_joint_state(
            'cephalothorax_joint',
            self.data.qpos[
                self.body_joint_qpos_adr:(self.body_joint_qpos_adr + 7)])
        leg_poses = []
        for leg_name in self.leg_names:
            qposes = self.legs[leg_name].get_qposes()
            leg_pose = LegPose()
            leg_pose.leg_name = leg_name
            leg_pose.coxa_qpos = qposes[0]
            leg_pose.femur_qpos = qposes[1]
            leg_pose.tibia_qpos = qposes[2]
            leg_poses.append(leg_pose)
        msg.leg_poses = leg_poses
        return msg

    def convert_vector3_to_list(self, vector3):
        """Convert geometry_msg Vector3 to a Python list."""
        return [vector3.x, vector3.y, vector3.z]

    def set_leg_targets_callback(self, msg):
        """Determine the poses required to best reach the targets."""
        leg_target_values = msg.leg_targets
        leg_targets = dict(zip(self.leg_names, leg_target_values))
        for leg_name in self.leg_names:
            # Move the claw to the target point
            self.legs[leg_name].move_claw_to_cartesian(
                self.convert_vector3_to_list(leg_targets[leg_name]))
        mujoco.mj_step(self.model, self.data)

        pose_msg = self.construct_pose_msg()
        self.spiderbot_target_pose_publisher.publish(pose_msg)
