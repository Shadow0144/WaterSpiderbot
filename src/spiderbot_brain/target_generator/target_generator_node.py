"""Generate training targets."""

import math
import random

from rclpy.node import Node

from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.msg import Target
from spiderbot_interfaces.srv import GetTrainingConfiguration

from std_msgs.msg import Empty
from std_msgs.msg import Int32

from std_srvs.srv import SetBool
from std_srvs.srv import Trigger


class TargetGeneratorNode(Node):
    """A node for generating training targets for a Spiderbot."""

    def __init__(self):
        """Initialize and run a brain."""
        super().__init__('target_generator_node')

        self.get_logger().info(
            'Starting spiderbot training target generator node'
        )

        self.last_timestamp = -1

        self.training_started = False

        self.waiting_for_simulation_reset = True
        self.time_to_reach_target_s = 10.0
        self.time_left_to_reach_target_s = 0.0
        self.num_targets_per_episodes = 30
        self.num_episodes_per_candidate = 10
        self.num_candidates_per_generation = 3
        self.num_targets_remaining = 0
        self.distance_scaling = 0.10
        self.rotation_half_range = math.pi / 16.0

        self.spiderbot_body_x = 0.0
        self.spiderbot_body_y = 0.0
        self.spiderbot_yaw = 0.0

        self.new_episode = True
        self.episode_direction = 0.0
        self.episode_rotation = 0.0

        self.create_new_target_on_reaching = False

        self.direction_jitter_half_range = math.pi / 16.0
        self.rotation_jitter_half_range = math.pi / 32.0

        self.target_publisher = self.create_publisher(
            Target,
            'target',
            10
        )

        self.target_number_publisher = self.create_publisher(
            Int32,
            'target_number',
            10
        )

        self.start_training_episode_publisher = self.create_publisher(
            Empty,
            'start_training_episode',
            10
        )

        self.spiderbot_pose_subscription = self.create_subscription(
            SpiderbotPose,
            'spiderbot_pose',
            self.spiderbot_pose_callback,
            10
        )

        self.target_reached_subscription = self.create_subscription(
            Empty,
            'target_reached',
            self.target_reached_callback,
            10
        )

        self.training_episode_terminated_subscription = (
            self.create_subscription(
                Empty,
                'training_episode_terminated',
                self.training_episode_terminated_callback,
                10
            )
        )

        self.get_training_configuration_service = self.create_service(
            GetTrainingConfiguration,
            'get_training_configuration',
            self.get_training_configuration_callback
        )

        self.enable_training_service = self.create_service(
            SetBool,
            'enable_training',
            self.enable_training_callback
        )

        self.reset_simulation_client = self.create_client(
            Trigger,
            'reset_simulation'
        )
        while not self.reset_simulation_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                'Waiting on reset_simulation service',
                once=True
            )
        self.get_logger().info('Simulation connected')

        self.get_logger().info('Spiderbot target generate node started')

    def is_running(self):
        """Return if the node is running."""
        return True

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

        # Return early if the training has not started
        #  (i.e. the locomotion node is not ready)
        # or the simulation has been requested to reset but has not finished
        if not self.training_started or self.waiting_for_simulation_reset:
            return

        # Get the time between the last poses and
        # decide to update the training target if too much time has passed
        delta_time = self._get_delta_time_from_timestamp(msg)
        self.time_left_to_reach_target_s -= delta_time
        if self.time_left_to_reach_target_s <= 0.0:
            self._generate_target()

    def target_reached_callback(self, msg):
        """Handle when the Spiderbot reaches the training target."""
        if self.create_new_target_on_reaching:
            self._generate_target()

    def training_episode_terminated_callback(self, msg):
        """Handle when the Spiderbot terminates the training episode."""
        # Set the number of targets remaining to 0 to trigger a simulation
        # reset when generating the next target
        self.num_targets_remaining = 0
        self._generate_target()

    def _generate_target(self):
        """Create a training target near the Spiderbot and publish it."""
        if self.num_targets_remaining <= 0:
            request = Trigger.Request()
            future = self.reset_simulation_client.call_async(request)
            future.add_done_callback(self.reset_simulation_callback)
            self.start_training_episode_publisher.publish(Empty())
            self.num_targets_remaining = self.num_targets_per_episodes
            self.waiting_for_simulation_reset = True
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

            self._set_target(target)
            self.time_left_to_reach_target_s = self.time_to_reach_target_s

    def get_training_configuration_callback(self, request, response):
        """Provide the training configuration."""
        response.training_mode_enabled = True
        response.num_targets_per_episode = self.num_targets_per_episodes
        response.num_episodes_per_candidate = self.num_episodes_per_candidate
        response.num_candidates_per_generation = (
            self.num_candidates_per_generation
        )
        return response

    def enable_training_callback(self, request, response):
        """Enable the training when the locomotion node is up and ready."""
        self.training_started = request.data
        self._generate_target()
        response.success = True
        return response

    def reset_simulation_callback(self, future):
        """Publish a new training target for the new episode."""
        self.waiting_for_simulation_reset = False
        self._generate_target()

    def _set_target(self, target):
        """Publish a new training target."""
        target_message = Target()
        target_message.target_x = target[0]
        target_message.target_y = target[1]
        target_message.target_theta = target[2]
        self.target_publisher.publish(target_message)
        target_num = (
            self.num_targets_per_episodes - self.num_targets_remaining
        )
        target_number_message = Int32()
        target_number_message.data = target_num
        self.target_number_publisher.publish(target_number_message)
        self.get_logger().info(
            f'Setting training target '
            f'({target_num}/{self.num_targets_per_episodes}): '
            f'{[f"{item:.6f}" for item in target]}'
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
