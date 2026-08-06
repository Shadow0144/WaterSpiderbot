"""Spiderbot locomotion node."""

from rcl_interfaces.msg import SetParametersResult

import rclpy
from rclpy.node import Node

from spiderbot_interfaces.msg import LegTargets
from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.msg import SpiderbotTargetPose
from spiderbot_interfaces.srv import GetSpiderbotDescription

from std_srvs.srv import Empty

from .modules import HandcraftedAngleModule
from .modules import HandcraftedPointModule
from .modules import SimpleSinModule


class SpiderbotLocomotionNode(Node):
    """Spiderbot locomotion."""

    def __init__(self):
        """Initialize and run a Spiderbot locomotor."""
        super().__init__('locomotion_node')

        self.declare_parameter('locomotion_module',
                               'handcrafted_point')
        self.locomotion_module_type = (
            self.get_parameter('locomotion_module').value
        )

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

        # Set the module after getting the description
        self.set_locomotion_module()
        self.add_on_set_parameters_callback(self.parameter_changed_callback)

        self.spiderbot_target_pose_publisher = self.create_publisher(
            SpiderbotTargetPose, 'spiderbot_target_pose', 10)

        self.leg_set_targets_publisher = self.create_publisher(
            LegTargets, 'set_leg_targets', 10)

        self.spiderbot_pose_subscription = self.create_subscription(
            SpiderbotPose,
            'spiderbot_pose',
            self.spiderbot_pose_callback,
            10
        )
        self.spiderbot_pose_subscription

        self.reset_simulation_client = self.create_client(
            Empty,
            'reset_simulation')

    def parameter_changed_callback(self, params):
        """React to parameters updating."""
        for param in params:
            if param.name == 'locomotion_module':
                self.locomotion_module_type = param.value
                self.set_locomotion_module()
        return SetParametersResult(successful=True)

    def set_locomotion_module(self):
        """Set the locomotion module."""
        if self.locomotion_module_type == 'simple_sin':
            self.locomotion_module = SimpleSinModule(self)
        elif self.locomotion_module_type == 'handcrafted_angle':
            self.locomotion_module = HandcraftedAngleModule(self)
        elif self.locomotion_module_type == 'handcrafted_point':
            self.locomotion_module = HandcraftedPointModule(self)

    def request_spiderbot_description(self):
        """Get the spec xml from the description."""
        request = GetSpiderbotDescription.Request()
        future = self.spiderbot_description_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

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

    def publish_angles(self, msg):
        """Publish target angles for the leg actuators."""
        self.spiderbot_target_pose_publisher.publish(msg)

    def publish_points(self, msg):
        """Publish target points for the leg to reach for."""
        self.leg_set_targets_publisher.publish(msg)

    def reset_simulation(self):
        """Request the simulation to reset."""
        request = Empty.Request()
        future = self.reset_simulation_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
