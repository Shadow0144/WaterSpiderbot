"""Spiderbot locomotion node."""

from rcl_interfaces.msg import SetParametersResult

import rclpy
from rclpy.node import Node

from spiderbot_interfaces.msg import LegTargets
from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.msg import SpiderbotTargetPose
from spiderbot_interfaces.msg import TrainingTarget
from spiderbot_interfaces.srv import GetSpiderbotDescription

from std_msgs.msg import Empty

from std_srvs.srv import SetBool


class LocomotionNode(Node):
    """Spiderbot locomotion."""

    def __init__(self, node_name):
        """Initialize and run a Spiderbot locomotor."""
        super().__init__(node_name)

        self.get_logger().info(
            f'Starting spiderbot locomotion node: {node_name}'
        )

        self.declare_parameter('training_mode_enabled',
                               True)
        self.training_mode_enabled = (
            self.get_parameter('training_mode_enabled').value
        )

        self.spiderbot_description_client = self.create_client(
            GetSpiderbotDescription,
            'get_spiderbot_description')
        while not self.spiderbot_description_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                'Waiting on get_spec_xml service',
                once=True)
        self.spiderbot_description = self.request_spiderbot_description()
        self.get_logger().info('Spiderbot description received')

        self.add_on_set_parameters_callback(self.parameter_changed_callback)

        self.spiderbot_target_pose_publisher = self.create_publisher(
            SpiderbotTargetPose,
            'spiderbot_target_pose',
            10
        )

        self.leg_set_targets_publisher = self.create_publisher(
            LegTargets,
            'set_leg_targets',
            10
        )

        self.training_target_reached_publisher = self.create_publisher(
            Empty,
            'training_target_reached',
            10
        )

        self.training_episode_terminated_publisher = self.create_publisher(
            Empty,
            'training_episode_terminated',
            10
        )

        self.spiderbot_pose_subscription = self.create_subscription(
            SpiderbotPose,
            'spiderbot_pose',
            self.spiderbot_pose_callback,
            10
        )

        self.training_target_subscription = self.create_subscription(
            TrainingTarget,
            'training_target',
            self.training_target_callback,
            10
        )

        self.start_training_episode_subscriber = self.create_subscription(
            Empty,
            'start_training_episode',
            self.start_training_episode_callback,
            10
        )

        self.set_training_mode_enabled_service = self.create_service(
            SetBool,
            'set_training_mode_enabled',
            self.set_training_mode_enabled_callback
        )

        self.get_logger().info('Spiderbot locomotion node started')

    def is_running(self):
        """Return if the node is running or if it's ready to shut down."""
        return True

    def parameter_changed_callback(self, params):
        """React to parameters updating."""
        for param in params:
            if param.name == 'training_mode_enabled':
                if self.locomotion_module is not None:
                    self.locomotion_module.set_training_mode_enabled(
                        param.value
                    )
        return SetParametersResult(successful=True)

    def request_spiderbot_description(self):
        """Get the spec xml from the description."""
        request = GetSpiderbotDescription.Request()
        future = self.spiderbot_description_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def spiderbot_pose_callback(self, msg):
        """Publish a set of leg targets whenever a new pose is received."""
        if self.locomotion_module is not None:
            self.locomotion_module.update(msg)

    def target_location_callback(self, msg):
        """Set the target for the locomotion module to approach."""
        if self.locomotion_module is not None:
            self.locomotion_module.set_target(msg)

    def publish_angles(self, msg):
        """Publish target angles for the leg actuators."""
        self.spiderbot_target_pose_publisher.publish(msg)

    def publish_points(self, msg):
        """Publish target points for the leg to reach for."""
        self.leg_set_targets_publisher.publish(msg)

    def publish_training_target_reached(self):
        """Publish that the training target was reached."""
        self.training_target_reached_publisher.publish(Empty())

    def publish_training_episode_terminated(self):
        """Publish that the training episode was terminated early."""
        self.training_episode_terminated_publisher.publish(Empty())

    def training_target_callback(self, msg):
        """Reset the simuation and has the Spiderbot move to the target."""
        if self.locomotion_module is not None:
            self.locomotion_module.set_training_target(msg)

    def start_training_episode_callback(self, msg):
        """Start a new training episode."""
        self.locomotion_module.start_training_episode()

    def set_training_mode_enabled_callback(self, request, response):
        """Toggle if training mode is enabled."""
        self.locomotion_module.set_training_mode_enabled(request.data)
        response.success = True
        response.message = 'Success'
        return response
