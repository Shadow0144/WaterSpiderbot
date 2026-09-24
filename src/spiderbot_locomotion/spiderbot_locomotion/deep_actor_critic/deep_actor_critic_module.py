"""A locomotion module using a Deep Neural Network."""

import time

import spiderbot_utilities as utils

from .policy.policy_trainer import PolicyTrainer
from ..locomotion_module import LocomotionModule


class DeepActorCriticModule(LocomotionModule):
    """A locomotion module using a Deep Neural Network Actor-Critic policy."""

    def __init__(
            self,
            locomotion_node,
            spiderbot_description,
            training_mode_enabled=True,
            use_population_training=True,
            use_soft_actor_critics_policy=True):
        """Initialize the locomotion module."""
        super().__init__(locomotion_node, spiderbot_description)

        self.training = training_mode_enabled
        self.using_population_training = (
            self.training and use_population_training
        )

        self.target = None
        self.episode_number = 0
        self.episode_save_interval = 10
        self.population_size = 10 if self.using_population_training else 1

        self.episode_terminated = False

        self.policy_trainer = PolicyTrainer(
            self.locomotion_node.get_logger(),
            use_soft_actor_critics_policy=use_soft_actor_critics_policy
        )
        self.are_poses_normalized = (
            self.policy_trainer.get_are_poses_normalized()
        )
        self.policy_trainer.load_population_checkpoint()

        # If not training, then just use the policy directly
        # The policy trainer can handle loading the policy first though
        if not self.training:
            self.policy = self.policy_trainer.policy

    def update(self, spiderbot_pose_msg):
        """Walk the spiderbot towards its target."""
        super().update(spiderbot_pose_msg)
        if self.delta_time <= 0.0:
            # Wait at least one step before doing anything
            return

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
                spiderbot_pose_msg
            )
            if angles is not None:
                self.publish_angles(angles)
            if training_status is not None:
                self.publish_training_status(training_status)

    def train_step(self, spiderbot_pose_msg):
        """Perform a single training step."""
        action_t, training_status = self.policy_trainer.train_step(
                spiderbot_pose_msg,
                self.delta_time
        )
        training_status.using_population_training = (
            self.using_population_training
        )

        self.locomotion_node.publish_training_status(training_status)

        if training_status.target_reached:
            self.locomotion_node.get_logger().info('Target reached')
            self.locomotion_node.publish_target_reached()

        self.episode_terminated = training_status.episode_terminated
        if self.episode_terminated:
            self.locomotion_node.get_logger().info(
                'Training episode terminated'
            )
            self.locomotion_node.publish_training_episode_terminated()

        return action_t, training_status

    def publish_angles(self, target_angles):
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
        self.locomotion_node.publish_angles(msg)

    def publish_training_status(self, training_status):
        """Publish information on the training step."""
        self.locomotion_node.publish_training_status(training_status)

    def set_target(self, set_target_msg):
        """Update the target."""
        super().set_target(set_target_msg)
        self.policy_trainer.set_target(self.target)

    def start_training_episode(self):
        """Start a new training episode."""
        self.reset()

    def reset(self):
        """Reset the neural network."""
        self.episode_terminated = False

        self.policy_trainer.start_new_training_episode()

        # Save the weights periodically
        self.episode_number += 1
        if self.episode_number % self.episode_save_interval == 1:
            if self.using_population_training:
                self.policy_trainer.save_population_checkpoint()
            else:
                self.policy.save_weights()

        if self.using_population_training:
            self.policy_trainer.set_target(
                self.target
            )
        else:
            self.policy.set_target(
                self.target
            )

        super().reset()

    def reset_learned_weights(self):
        """Back up the current weights and start with new random weights."""
        if self.training:
            self.policy_trainer.reset_checkpoints()
        else:
            self.policy.reset_learned_weights()
