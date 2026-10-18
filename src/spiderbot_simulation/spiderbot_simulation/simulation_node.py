"""Simulate a Spiderbot and launch an interactive viewer."""

import time
import os
from functools import partial

import mujoco

import rclpy
from rclpy.node import Node

from spiderbot_interfaces.msg import LegTargets
from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.msg import SpiderbotTargetPose
from spiderbot_interfaces.msg import Target
from spiderbot_interfaces.msg import TrainingStatus
from spiderbot_interfaces.srv import GetSpiderbotDescription
from spiderbot_interfaces.srv import GetTrainingConfiguration

from std_msgs.msg import Int32

from std_srvs.srv import Trigger

from .simulated_spiderbot import SimulatedSpiderbot
from .simulation_viewer import SimulationViewer


class SimulationNode(Node):
    """A simulation node for a Spiderbot."""

    def __init__(self):
        """Initialize and run a simulation."""
        super().__init__('simulation_node')

        self.get_logger().info('Starting spiderbot simulation node')

        target_publish_rate_ps = 60.0
        self.publish_interval = 1.0 / target_publish_rate_ps
        self.last_timestamp = time.time()

        self.training_mode_enabled = False
        self.num_targets_per_episode = 0
        self.num_episodes_per_candidate = 0
        self.num_candidates_per_generation = 0

        self.simulated_spiderbots = {}
        self.candidate_topic_prefixes = []
        self.topic_prefixes = []

        self.reset_simulation_service = self.create_service(
            Trigger,
            'reset_simulation',
            self.reset_simulation_callback
        )

        self.get_training_configuration_client = self.create_client(
            GetTrainingConfiguration,
            'get_training_configuration'
        )
        while not self.get_training_configuration_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                'Waiting on training configuration service',
                once=True
            )
        self._get_training_configuration()
        self.get_logger().info('Spiderbot training configuration received')

        self.spiderbot_description_client = self.create_client(
            GetSpiderbotDescription,
            'get_spiderbot_description')
        while not self.spiderbot_description_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                'Waiting on spiderbot description service',
                once=True)
        self.spiderbot_description = self._get_spiderbot_description()
        self.get_logger().info('Spiderbot description received')

        spec = mujoco.MjSpec.from_string(self.spiderbot_description.spec_xml)
        spec.compiler.meshdir = self.spiderbot_description.mesh_dir
        spec.compiler.texturedir = self.spiderbot_description.texture_dir
        model = spec.compile()
        self.timestep = model.opt.timestep

        self._create_spiderbot_topics()

        for topic_prefix in self.topic_prefixes:
            self.simulated_spiderbots[topic_prefix] = SimulatedSpiderbot(
                topic_prefix,
                self.spiderbot_description
            )

        self.viewer = SimulationViewer(
            self.get_logger(),
            self.topic_prefixes,
            self.simulated_spiderbots,
            self.train_parallel
        )
        if self.training_mode_enabled:
            self.viewer.set_training_configuration(
                self.num_targets_per_episode,
                self.num_episodes_per_candidate,
                self.num_candidates_per_generation
            )

        self.last_timestamp = time.time()

        self.get_logger().info('Spiderbot simulation node started')

    def destroy_node(self):
        """Destroy the window and finish destroying the node."""
        self.viewer.destroy()
        return super().destroy_node()

    def is_running(self):
        """Return if the node is running or if it's ready to shut down."""
        return self.viewer.is_running()

    def _get_spiderbot_description(self):
        """Get the spec xml from the description."""
        request = GetSpiderbotDescription.Request()
        future = self.spiderbot_description_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def _get_training_configuration(self):
        """Get the training configuration from the brain."""
        request = GetTrainingConfiguration.Request()
        future = self.get_training_configuration_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        self.training_mode_enabled = result.training_mode_enabled
        self.num_targets_per_episode = result.num_targets_per_episode
        self.num_episodes_per_candidate = result.num_episodes_per_candidate
        self.candidate_topic_prefixes = result.candidate_topic_prefixes
        self.train_parallel = result.train_parallel
        self.num_candidates_per_generation = (
            len(self.candidate_topic_prefixes)
        )
        self.topic_prefixes = (
            self.candidate_topic_prefixes
            if self.train_parallel else
            ['']
        )

    def _create_spiderbot_topics(self):
        """Create the topics for each spiderbot."""
        self.spiderbot_pose_publishers = {}
        self.set_leg_targets_subscriptions = {}
        self.spiderbot_target_pose_subscriptions = {}
        self.target_subscriptions = {}
        self.target_number_subscriptions = {}
        self.training_status_subscriptions = {}
        for topic_prefix in self.topic_prefixes:
            self.spiderbot_pose_publishers[topic_prefix] = (
                self.create_publisher(
                    SpiderbotPose,
                    f'{topic_prefix}spiderbot_pose',
                    10
                )
            )

            set_leg_targets_callback_full = (
                partial(self.set_leg_targets_callback, topic_prefix)
            )
            self.set_leg_targets_subscriptions[topic_prefix] = (
                self.create_subscription(
                    LegTargets,
                    f'{topic_prefix}set_leg_targets',
                    set_leg_targets_callback_full,
                    10
                )
            )

            spiderbot_target_pose_callback_full = (
                partial(self.spiderbot_target_pose_callback, topic_prefix)
            )
            self.spiderbot_target_pose_subscriptions[topic_prefix] = (
                self.create_subscription(
                    SpiderbotTargetPose,
                    f'{topic_prefix}spiderbot_target_pose',
                    spiderbot_target_pose_callback_full,
                    10
                )
            )

            target_callback_full = (
                partial(self.target_callback, topic_prefix)
            )
            self.target_subscriptions[topic_prefix] = (
                self.create_subscription(
                    Target,
                    f'{topic_prefix}target',
                    target_callback_full,
                    10
                )
            )

            target_number_callback_full = (
                partial(self.target_number_callback, topic_prefix)
            )
            self.target_number_subscriptions[topic_prefix] = (
                self.create_subscription(
                    Int32,
                    f'{topic_prefix}target_number',
                    target_number_callback_full,
                    10
                )
            )

            training_status_callback_full = (
                partial(self.training_status_callback, topic_prefix)
            )
            self.training_status_subscriptions[topic_prefix] = (
                self.create_subscription(
                    TrainingStatus,
                    f'{topic_prefix}training_status',
                    training_status_callback_full,
                    10
                )
            )

    def set_leg_targets_callback(self, topic_prefix, msg):
        """Move the mocaps to the targets."""
        if topic_prefix in self.simulated_spiderbots.keys():
            self.simulated_spiderbots[topic_prefix].set_leg_targets(msg)

    def spiderbot_target_pose_callback(self, topic_prefix, msg):
        """Update the targets for the actuators."""
        if topic_prefix in self.simulated_spiderbots.keys():
            self.simulated_spiderbots[topic_prefix].set_target_pose(msg)

    def target_callback(self, topic_prefix, msg):
        """Move the target to the location and make it visible."""
        if topic_prefix in self.simulated_spiderbots.keys():
            self.simulated_spiderbots[topic_prefix].set_target(msg)

    def target_number_callback(self, topic_prefix, msg):
        """Update the viewer with the current target number."""
        self.viewer.update_target_number(topic_prefix, msg.data)

    def training_status_callback(self, topic_prefix, msg):
        """Enable displaying and update the information on the training."""
        self.viewer.update_training_status(topic_prefix, msg)

    def reset_simulation_callback(self, request, response):
        """Reset simulation."""
        for spiderbot in self.simulated_spiderbots.values():
            spiderbot.reset()
        self._publish_poses()
        response.success = True
        return response

    def _publish_poses(self):
        """Publish the current pose."""
        for topic_prefix in self.topic_prefixes:
            if (
                topic_prefix in self.simulated_spiderbots.keys() and
                topic_prefix in self.spiderbot_pose_publishers.keys()
            ):
                spiderbot = self.simulated_spiderbots[topic_prefix]
                self.spiderbot_pose_publishers[topic_prefix].publish(
                    spiderbot.create_pose_message(
                        self.last_timestamp
                    )
                )

    def update(self):
        """Step the physics, publish the pose, and update the render."""
        if not self.viewer.is_running():
            return

        current_timestamp = time.time()

        # Step the physics
        for spiderbot in self.simulated_spiderbots.values():
            spiderbot.step()

        # Publish the current pose if enough time has elapsed
        if current_timestamp - self.last_timestamp >= self.publish_interval:
            self.last_timestamp = current_timestamp
            self._publish_poses()

        # Update the renderer
        self.viewer.update(current_timestamp)

        # Sleep until it is time for the next simulation step
        time_elapsed = time.time() - current_timestamp
        time_until_next_step = (
            self.timestep - time_elapsed
        )
        if time_until_next_step > 0.0:
            time.sleep(time_until_next_step)
