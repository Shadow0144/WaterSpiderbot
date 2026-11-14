"""Provide a teleop interface for a Spiderbot."""

from rclpy.node import Node

from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.msg import Target
from spiderbot_interfaces.srv import GetTrainingConfiguration

from .controller import Controller


class TeleopNode(Node):
    """A teleop node for a Spiderbot."""

    def __init__(self):
        """Initialize the teleop node."""
        super().__init__('teleop_node')

        self.controller = Controller(self)

        self.target_publisher = self.create_publisher(
            Target,
            'target',
            10
        )

        self.spiderbot_pose_subscription = self.create_subscription(
            SpiderbotPose,
            'spiderbot_pose',
            self.spiderbot_pose_callback,
            10
        )

        self.get_training_configuration_service = self.create_service(
            GetTrainingConfiguration,
            'get_training_configuration',
            self.get_training_configuration_callback
        )

    def destroy_node(self):
        """Destroy the window and finish destroying the node."""
        self.controller.destroy()
        return super().destroy_node()

    def get_is_running(self):
        """Return if the node is running."""
        return self.controller.get_is_running()

    def get_training_configuration_callback(self, request, response):
        """Provide the training configuration."""
        response.training_mode_enabled = False
        response.num_targets_per_episode = 0
        response.num_episodes_per_candidate = 0
        response.candidate_topic_prefixes = ['spiderbot']
        response.train_parallel = False
        return response

    def set_target(self, target):
        """Publish a new target."""
        msg = Target()
        msg.target_x = target[0]
        msg.target_y = target[1]
        msg.target_theta = target[2]
        self.target_publisher.publish(msg)
        self.get_logger().info(
            f'Setting target '
            f'{[f"{item:.6f}" for item in target]}',
            throttle_duration_sec=3.0
        )

    def spiderbot_pose_callback(self, msg):
        """Handle the updated Spiderbot pose."""
        # Update the stored position of the Spiderbot
        pose = msg.body_odometry.pose.pose
        self.controller.update_spiderbot_pose(pose)

    def update(self):
        """Update the node and viewer."""
        if not self.get_is_running():
            return

        self.controller.update()
