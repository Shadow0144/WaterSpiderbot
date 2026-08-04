"""Spiderbot locomotion node."""

from geometry_msgs.msg import Vector3

import rclpy
from rclpy.node import Node

from spiderbot_interfaces.msg import LegTargets
from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.srv import GetSpiderbotDescription

from . import MoveToPointLocomotionModule


class SpiderbotLocomotionNode(Node):
    """Spiderbot locomotion."""

    def __init__(self):
        """Initialize and run a Spiderbot locomotor."""
        super().__init__('locomotion_node')

        self.locomotion_module = MoveToPointLocomotionModule()

        self.last_timestamp = -1.0

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

        self.leg_set_targets_publisher = self.create_publisher(
            LegTargets, 'set_leg_targets', 10)

        self.spiderbot_pose_subscription = self.create_subscription(
            SpiderbotPose,
            'spiderbot_pose',
            self.spiderbot_pose_callback,
            10
        )
        self.spiderbot_pose_subscription

    def request_spiderbot_description(self):
        """Get the spec xml from the description."""
        request = GetSpiderbotDescription.Request()
        future = self.spiderbot_description_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def create_vector3(self, target):
        """Construct a Vector3."""
        vector3 = Vector3(
            x=target[0],
            y=target[1],
            z=target[2]
        )
        return vector3

    def spiderbot_pose_callback(self, msg):
        """Publish a set of leg targets whenever a new pose is received."""
        if (self.last_timestamp < 0.0):
            # Skip the first update to make sure we have an
            # appropriate delta time
            self.last_timestamp = msg.timestamp
        else:
            delta_time = msg.timestamp - self.last_timestamp
            self.last_timestamp = msg.timestamp
            self.locomotion_module.walk_forward(delta_time)
            msg = LegTargets()
            leg_targets = []
            for leg_name in self.leg_names:
                current_target = (
                    self.locomotion_module.current_targets[leg_name]
                )
                leg_target = self.create_vector3(current_target)
                leg_targets.append(leg_target)
            msg.leg_targets = leg_targets
            self.leg_set_targets_publisher.publish(msg)
