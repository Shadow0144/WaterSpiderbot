"""Spiderbot locomotion node using a deep learning actor-critic."""

from functools import partial

import rclpy
from rclpy.node import Node

from spiderbot_interfaces.msg import LegTargets
from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.msg import SpiderbotTargetPose
from spiderbot_interfaces.msg import Target
from spiderbot_interfaces.msg import TrainingStatus
from spiderbot_interfaces.srv import GetSpiderbotDescription
from spiderbot_interfaces.srv import GetTrainingConfiguration

from std_msgs.msg import Bool
from std_msgs.msg import Empty

from std_srvs.srv import Trigger

from .deep_actor_critic_module import DeepActorCriticModule


class DeepActorCriticLocomotionNode(Node):
    """Spiderbot locomotion using a deep learning actor-critic."""

    def __init__(self):
        """Initialize and run a Spiderbot locomotor."""
        super().__init__('deep_actor_critic_locomotion_node')

        self.get_logger().info(
            'Starting spiderbot locomotion node'
        )

        self.locomotion_module = None
        self.training_mode_enabled = False
        self.num_targets_per_episode = 0
        self.num_episodes_per_candidate = 0
        self.num_candidates_per_generation = 0

        self.spiderbot_description_client = self.create_client(
            GetSpiderbotDescription,
            'get_spiderbot_description'
        )
        while not self.spiderbot_description_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                'Waiting on get_spec_xml service',
                once=True
            )
        self._get_spiderbot_description()
        self.get_logger().info('Spiderbot description received')

        self.get_training_configuration_client = self.create_client(
            GetTrainingConfiguration,
            'get_training_configuration'
        )
        while not self.get_training_configuration_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                'Waiting on get_training_configuration service',
                once=True
            )
        self.get_logger().info('Spiderbot training configuration received')
        self._get_training_configuration()

        self._create_spiderbot_topics()

        self.start_training_episode_service = self.create_service(
            Trigger,
            'start_training_episode',
            self.start_training_episode_callback
        )

        self.start_training_client = self.create_client(
            Trigger,
            'start_training'
        )

        # Set the module after getting the description
        self.locomotion_module = DeepActorCriticModule(
            self,
            self.spiderbot_description,
            self.training_mode_enabled,
            self.num_episodes_per_candidate,
            self.candidate_topic_prefixes,
            self.train_parallel
        )

        self.training_status_publishers = {}
        if self.train_parallel:
            self.topic_prefixes = self.candidate_topic_prefixes
        else:
            self.topic_prefixes = ['']
        for topic_prefix in self.topic_prefixes:
            self.training_status_publishers[topic_prefix] = (
                self.create_publisher(
                    TrainingStatus,
                    f'{topic_prefix}training_status',
                    10
                )
            )

        self.reset_learned_weights_service = self.create_service(
            Trigger,
            'reset_learned_weights',
            self.reset_learned_weights_callback
        )

        if self.training_mode_enabled:
            start_training_request = Trigger.Request()
            self.start_training_client.call_async(start_training_request)

        self.get_logger().info('Spiderbot locomotion node started')

    def is_running(self):
        """Return if the node is running or if it's ready to shut down."""
        return True

    def _get_spiderbot_description(self):
        """Get the spec xml from the description."""
        request = GetSpiderbotDescription.Request()
        future = self.spiderbot_description_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        self.spiderbot_description = result

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
        self.num_candidates_per_generation = (
            len(self.candidate_topic_prefixes)
        )
        self.train_parallel = result.train_parallel
        # If training in parallel, each policy has its own nested namespace
        # If training in series, there is only the top-level namespace
        if self.train_parallel:
            self.topic_prefixes = result.candidate_topic_prefixes
        else:
            self.topic_prefixes = ['']

    def _create_spiderbot_topics(self):
        """Create all the topics unique to each spiderbot."""
        self.spiderbot_target_pose_publishers = {}
        self.leg_set_targets_publishers = {}
        self.target_reached_publishers = {}
        self.training_episode_terminated_publishers = {}
        self.spiderbot_pose_subscriptions = {}
        self.target_subscriptions = {}
        for topic_prefix in self.topic_prefixes:
            self.spiderbot_target_pose_publishers[topic_prefix] = (
                self.create_publisher(
                    SpiderbotTargetPose,
                    f'{topic_prefix}spiderbot_target_pose',
                    10
                )
            )

            self.leg_set_targets_publishers[topic_prefix] = (
                self.create_publisher(
                    LegTargets,
                    f'{topic_prefix}set_leg_targets',
                    10
                )
            )

            self.target_reached_publishers[topic_prefix] = (
                self.create_publisher(
                    Bool,
                    f'{topic_prefix}target_reached',
                    10
                )
            )

            self.training_episode_terminated_publishers[topic_prefix] = (
                self.create_publisher(
                    Empty,
                    f'{topic_prefix}training_episode_terminated',
                    10
                )
            )

            spiderbot_pose_callback_full = (
                partial(self.spiderbot_pose_callback, topic_prefix)
            )
            self.spiderbot_pose_subscriptions[topic_prefix] = (
                self.create_subscription(
                    SpiderbotPose,
                    f'{topic_prefix}spiderbot_pose',
                    spiderbot_pose_callback_full,
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

    def start_training_episode_callback(self, request, response):
        """Start a new training episode."""
        self.locomotion_module.start_training_episode()
        response.success = True
        return response

    def spiderbot_pose_callback(self, topic_prefix, msg):
        """Publish a set of leg targets whenever a new pose is received."""
        if self.locomotion_module is not None:
            self.locomotion_module.update(topic_prefix, msg)

    def target_callback(self, topic_prefix, msg):
        """Reset the simuation and has the Spiderbot move to the target."""
        if self.locomotion_module is not None:
            self.locomotion_module.set_target(topic_prefix, msg)

    def publish_target_pose(self, topic_prefix, msg):
        """Publish target angles for the leg actuators."""
        self.spiderbot_target_pose_publishers[topic_prefix].publish(msg)

    def publish_target_reached(self, topic_prefix, target_reached):
        """Publish if the training target was reached."""
        msg = Bool()
        msg.data = target_reached
        self.target_reached_publishers[topic_prefix].publish(msg)

    def publish_training_episode_terminated(self, topic_prefix):
        """Publish that the training episode was terminated early."""
        msg = Empty()
        self.training_episode_terminated_publishers[topic_prefix].publish(
            msg
        )

    def publish_training_status(self, topic_prefix, training_status):
        """Publish information on the training and the reward."""
        msg = TrainingStatus()
        msg.step_reward = training_status.step_reward
        msg.episode_reward = training_status.episode_reward
        msg.candidate_reward = training_status.candidate_reward
        msg.using_population_training = (
            training_status.using_population_training
        )
        msg.episode_number = training_status.episode_number
        msg.episodes_per_candidate = training_status.episodes_per_candidate
        msg.candidate_number = training_status.candidate_number
        msg.candidates_per_generation = (
            training_status.candidates_per_generation
        )
        msg.generation_number = training_status.generation_number
        msg.target_reached = training_status.target_reached
        msg.episode_terminated = training_status.episode_terminated
        msg.reward_component_labels = (
            training_status.reward_component_labels
        )
        msg.reward_component_values = (
            training_status.reward_component_values
        )
        self.training_status_publishers[topic_prefix].publish(msg)

    def reset_learned_weights_callback(self, request, response):
        """Backup the current weights and start with new random weights."""
        self.locomotion_module.reset_learned_weights()
        response.success = True
        response.message = 'Success'
        return response
