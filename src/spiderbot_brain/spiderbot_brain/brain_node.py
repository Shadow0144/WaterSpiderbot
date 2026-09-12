"""Perform high-level planning and training."""

import math
import random

import rclpy
from rclpy.node import Node

from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.msg import TrainingTarget

from std_msgs.msg import Empty as EmptyMsg

from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import SetBool


class BrainNode(Node):
    """A brain node for a Spiderbot."""

    def __init__(self):
        """Initialize and run a brain."""
        super().__init__('brain_node')

        self.get_logger().info('Starting spiderbot brain node')

        self.last_timestamp = -1

        self.time_to_reach_target_s = 10
        self.time_left_to_reach_target_s = self.time_to_reach_target_s
        self.num_targets_per_episodes = 30
        self.num_targets_remaining = 0
        self.distance_scaling = 0.10
        self.rotation_half_range = math.pi / 16.0

        self.spiderbot_body_x = 0.0
        self.spiderbot_body_y = 0.0
        self.spiderbot_yaw = 0.0

        self.new_episode = True
        self.episode_direction = 0.0
        self.episode_rotation = 0.0

        self.direction_jitter_half_range = math.pi / 16
        self.rotation_jitter_half_range = math.pi / 32

        self.declare_parameter('training_mode_enabled',
                               True)
        self.training_mode_enabled = (
            self.get_parameter('training_mode_enabled').value
        )

        self._set_training_mode_enabled_client = self.create_client(
            SetBool,
            'set_training_mode_enabled')
        while not self._set_training_mode_enabled_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                'Waiting on set_training_mode_enabled service',
                once=True
            )
        _ = self._set_training_mode()
        self.get_logger().info('Training mode status set')

        self.training_target_publisher = self.create_publisher(
            TrainingTarget,
            'training_target',
            10
        )

        self.start_training_episode_publisher = self.create_publisher(
            EmptyMsg,
            'start_training_episode',
            10
        )

        self.spiderbot_pose_subscription = self.create_subscription(
            SpiderbotPose,
            'spiderbot_pose',
            self.spiderbot_pose_callback,
            10
        )

        self.training_target_reached_subscription = self.create_subscription(
            EmptyMsg,
            'training_target_reached',
            self.training_target_reached_callback,
            10
        )

        self.training_episode_terminated_subscription = (
            self.create_subscription(
                EmptyMsg,
                'training_episode_terminated',
                self.training_episode_terminated_callback,
                10
            )
        )

        self.reset_simulation_client = self.create_client(
            EmptySrv,
            'reset_simulation'
        )

        self.get_logger().info('Spiderbot brain node started')

    def spiderbot_pose_callback(self, msg):
        """Handle the updated Spiderbot pose."""
        # Update the stored position of the Spiderbot
        pose = msg.body_odometry.pose.pose
        # Position
        self.spiderbot_body_x = pose.position.x
        self.spiderbot_body_y = pose.position.y
        # Angle
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w
        self.spiderbot_yaw = (
            math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        )

        # Get the time between the last poses
        # and decide to update the training target if too much time has passed
        delta_time = self._get_delta_time_from_timestamp(msg)
        self.time_left_to_reach_target_s -= delta_time
        if self.time_left_to_reach_target_s <= 0.0:
            self._generate_training_target()

    def training_target_reached_callback(self, msg):
        """Handle when the Spiderbot reaches the training target."""
        self._generate_training_target()

    def training_episode_terminated_callback(self, msg):
        """Handle when the Spiderbot terminates the training episode."""
        # Set the number of targets remaining to 0 to trigger a simulation
        # reset when generating the next target
        self.num_targets_remaining = 0
        self._generate_training_target()

    def _set_training_mode(self):
        """Call the service to set the training mode."""
        request = SetBool.Request()
        future = self._set_training_mode_enabled_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def _generate_training_target(self):
        """Create a training target near the Spiderbot and publish it."""
        if self.num_targets_remaining <= 0:
            request = EmptySrv.Request()
            self.reset_simulation_client.call_async(request)
            self.start_training_episode_publisher.publish(EmptyMsg())
            self.num_targets_remaining = self.num_targets_per_episodes
            self.new_episode = True
        else:
            self.num_targets_remaining -= 1

            if self.new_episode:
                self.episode_direction = random.uniform(0.0, 2.0 ** math.pi)
                self.episode_rotation = (
                    random.uniform(
                        -self.rotation_half_range, self.rotation_half_range
                    )
                )
                self.new_episode = False

            direction = self.episode_direction + random.uniform(
                    -self.direction_jitter_half_range,
                    self.direction_jitter_half_range
                )
            direction_x = math.cos(direction) * self.distance_scaling
            direction_y = math.sin(direction) * self.distance_scaling
            target_x = (
                self.spiderbot_body_x + direction_x
            )
            target_y = (
                self.spiderbot_body_y + direction_y
            )

            rotation = self.episode_rotation + (
                random.uniform(
                    -self.rotation_jitter_half_range,
                    self.rotation_jitter_half_range
                )
            )
            target_angle = self.spiderbot_yaw + rotation

            target = [
                target_x,
                target_y,
                target_angle
            ]

            self._set_training_target(target)
            self.time_left_to_reach_target_s = self.time_to_reach_target_s

    def _set_training_target(self, target):
        """Publish a new training target."""
        msg = TrainingTarget()
        msg.target_x = target[0]
        msg.target_y = target[1]
        msg.target_theta = target[2]
        self.training_target_publisher.publish(msg)
        target_num = (
            self.num_targets_per_episodes - self.num_targets_remaining
        )
        self.get_logger().info(
            f'Setting training target '
            f'({target_num}/{self.num_targets_per_episodes}): {target}'
        )

    def _get_delta_time_from_timestamp(self, spiderbot_pose_msg):
        """Get the change in time between messages."""
        if self.last_timestamp < 0.0:
            # Skip the first update to make sure we have an
            # appropriate delta time
            self.last_timestamp = spiderbot_pose_msg.timestamp
            return 0.0
        else:
            delta_time = spiderbot_pose_msg.timestamp - self.last_timestamp
            self.last_timestamp = spiderbot_pose_msg.timestamp
            return delta_time

    def update(self):
        """Update the node."""
        # The node updates via callbacks triggered by other nodes,
        # so there's nothing to do here
        pass
