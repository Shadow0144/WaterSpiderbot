"""A locomotion module using a Deep Neural Network."""

import time

import spiderbot_utilities as utils

from .policy.parallel_policy_trainer import ParallelPolicyTrainer
from .policy.series_policy_trainer import SeriesPolicyTrainer
from ..locomotion_module import LocomotionModule


class DeepActorCriticModule(LocomotionModule):
    """A locomotion module using a Deep Neural Network Actor-Critic policy."""

    def __init__(
            self,
            locomotion_node,
            spiderbot_description,
            training_mode_enabled,
            num_episodes_per_candidate,
            candidate_topic_prefixes,
            train_parallel,
            use_soft_actor_critics_policy=True):
        """Initialize the locomotion module."""
        super().__init__(locomotion_node, spiderbot_description)

        self.training = training_mode_enabled
        self.num_episodes_per_candidate = num_episodes_per_candidate
        self.candidate_topic_prefixes = candidate_topic_prefixes
        self.num_candidates_per_generation = len(self.candidate_topic_prefixes)
        self.train_parallel = self.training and train_parallel
        self.using_population_training = (
            self.training and self.num_episodes_per_candidate > 1
        )

        self.episode_terminated = False

        self.are_poses_normalized = True
        if self.train_parallel:
            self.policy_trainer = (
                ParallelPolicyTrainer(
                    self.locomotion_node.get_logger(),
                    self.candidate_topic_prefixes,
                    self.num_episodes_per_candidate,
                    use_soft_actor_critics_policy
                )
            )
        else:
            self.policy_trainer = (
                SeriesPolicyTrainer(
                    self.locomotion_node.get_logger(),
                    self.candidate_topic_prefixes,
                    self.num_episodes_per_candidate,
                    use_soft_actor_critics_policy
                )
            )

        self.are_poses_normalized = (
                self.policy_trainer.get_are_poses_normalized()
        )

        self.policy_trainer.load_population_checkpoint()

        # If not training, then just use the policy directly
        # The policy trainer can handle loading the policy first though
        if not self.training:
            self.policy = self.policy_trainers[''].policy

    def update(self, topic_prefix, spiderbot_pose_msg):
        """Walk the spiderbot towards its target."""
        super().update(topic_prefix, spiderbot_pose_msg)
        # if self.delta_time <= 0.0:
        #     # Wait at least one step before doing anything
        #     return

        if self.episode_terminated:
            # Wait until a new episode starts
            return

        if not self.training:
            angles = self.policy.select_action(
                spiderbot_pose_msg
            )
            if angles is not None:
                self.publish_angles(angles)
        else:
            angles, training_status = self.train_step(
                topic_prefix,
                spiderbot_pose_msg
            )
            if angles is not None:
                self.publish_angles(
                    topic_prefix,
                    angles
                )
            if training_status is not None:
                self.publish_training_status(
                    topic_prefix,
                    training_status
                )

    def train_step(self, topic_prefix, spiderbot_pose_msg):
        """Perform a single training step."""
        action_t, training_status = self.policy_trainer.train_step(
                topic_prefix,
                spiderbot_pose_msg,
                self.delta_time
        )

        if not training_status:
            return None, None

        training_status.using_population_training = (
            self.using_population_training
        )

        self.locomotion_node.publish_training_status(
            topic_prefix,
            training_status
        )

        if training_status.target_reached:
            self.locomotion_node.get_logger().info(
                'Target reached',
                throttle_duration_sec=1.0
            )
            self.locomotion_node.publish_target_reached(topic_prefix)

        self.episode_terminated = training_status.episode_terminated
        if self.episode_terminated:
            self.locomotion_node.get_logger().info(
                'Training episode terminated'
            )
            self.locomotion_node.publish_training_episode_terminated(
                topic_prefix
            )

        return action_t, training_status

    def publish_angles(self, topic_prefix, target_angles):
        """Publish target angles for the leg actuators."""
        target_angles_per_leg = {}
        for i, leg_name in enumerate(self.leg_names):
            index = i * 3
            target_angles_per_leg[leg_name] = (
                [float(x) for x in target_angles[index:index+3]]
            )
        msg = utils.construct_target_pose_msg(
                    time.time(),
                    self.are_poses_normalized,
                    self.leg_names,
                    target_angles_per_leg
                )
        self.locomotion_node.publish_angles(topic_prefix, msg)

    def publish_training_status(self, topic_prefix, training_status):
        """Publish information on the training step."""
        self.locomotion_node.publish_training_status(
            topic_prefix,
            training_status
        )

    def set_target(self, topic_prefix, set_target_msg):
        """Update the target."""
        target_x = set_target_msg.target_x
        target_y = set_target_msg.target_y
        target_theta = set_target_msg.target_theta
        target = [
            target_x,
            target_y,
            target_theta
        ]
        self.policy_trainer.set_target(topic_prefix, target)

    def start_training_episode(self):
        """Start a new training episode."""
        self.reset()

    def reset(self):
        """Reset the neural network."""
        self.episode_terminated = False
        self.policy_trainer.start_new_training_episode()
        super().reset()

    def reset_learned_weights(self):
        """Back up the current weights and start with new random weights."""
        if self.training:
            self.policy_trainer.reset_checkpoints()
        else:
            self.policy.reset_learned_weights()
