"""Perform high-level planning and training."""

import math
import random
import time

import rclpy
from rclpy.node import Node

from spiderbot_interfaces.msg import TrainingTarget

from std_srvs.srv import SetBool


class BrainNode(Node):
    """A brain node for a Spiderbot."""

    def __init__(self):
        """Initialize and run a brain."""
        super().__init__('brain_node')

        self.time_min_s = 10
        self.time_max_s = 20
        self.distance_scaling = 0.1
        self.final_angle_scaling = 0.01

        self.declare_parameter('training_mode_enabled',
                               True)
        self.training_mode_enabled = (
            self.get_parameter('training_mode_enabled').value
        )

        self.set_training_mode_enabled_client = self.create_client(
            SetBool,
            'set_training_mode_enabled')
        while not self.set_training_mode_enabled_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                'Waiting on set_training_mode_enabled service'
            )
        _ = self.set_training_mode()

        self.training_target_publisher = self.create_publisher(
            TrainingTarget,
            'training_target',
            10)

    def set_training_mode(self):
        """Call the service to set the training mode."""
        request = SetBool.Request()
        future = self.set_training_mode_enabled_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def perform_training_step(self):
        """Set times and targets for the locomotion module to aim for."""
        time_to_reach_goal_s = float(random.randint(
            self.time_min_s,
            self.time_max_s
        ))
        heading_angle = random.uniform(0.0, 2.0 ** math.pi)
        final_angle = math.pi + heading_angle + (
            self.final_angle_scaling * random.uniform(0.0, 2.0 ** math.pi)
        )
        distance_scaling = time_to_reach_goal_s * self.distance_scaling
        target = [
            math.cos(heading_angle) * distance_scaling,
            math.sin(heading_angle) * distance_scaling,
            final_angle]

        self.set_training_target(time_to_reach_goal_s, target)

        time.sleep(time_to_reach_goal_s)

    def set_training_target(self, time_to_reach_goal_s, target):
        """Publish a new training target."""
        msg = TrainingTarget()
        msg.time_to_reach_goal_s = time_to_reach_goal_s
        msg.target_x = target[0]
        msg.target_y = target[1]
        msg.target_theta = target[2]
        self.training_target_publisher.publish(msg)
