"""A locomotion module using a Deep Neural Network."""

import time

import spiderbot_utilities as utils

from .neural_network.deep_actor_critic_policy import DeepActorCriticPolicy
from .neural_network.population_trainer import PopulationTrainer
from ..locomotion_module import LocomotionModule


class DeepActorCriticModule(LocomotionModule):
    """A locomotion module using a Deep Neural Network Actor-Critic policy."""

    def __init__(self, locomotion_node, spiderbot_description):
        """Initialize the locomotion module."""
        super().__init__(locomotion_node, spiderbot_description)

        self.target = None

        self.training = True
        self.population_training = self.training and True
        self.num_episodes = 0
        self.episode_save_interval = 10

        if self.population_training:
            self.population_trainer = PopulationTrainer(
                self.locomotion_node.get_logger()
            )
            self.population_trainer.load_population_checkpoint()
        else:
            self.policy = DeepActorCriticPolicy(
                self.locomotion_node.get_logger()
            )
            self.policy.load_weights()

    def update(self, spiderbot_pose_msg):
        """Walk the spiderbot towards its target."""
        delta_time = self.get_delta_time_from_msg(spiderbot_pose_msg)
        if delta_time <= 0.0:
            # Wait at least one step before doing anything
            return

        if self.is_resetting:
            # Wait until the reset is complete
            return

        if not self.training:
            angles = self.policy.select_action(
                spiderbot_pose_msg
            )
        else:
            angles = self.train_step(
                spiderbot_pose_msg,
                delta_time
            )
        if angles is not None:
            self.publish_angles(angles)

    def train_step(self, spiderbot_pose_msg, delta_time):
        """Perform a single training step."""
        if self.population_training:
            action_np, reward, done = (
                self.population_trainer.train_step(
                    spiderbot_pose_msg,
                    delta_time
                )
            )
        else:
            action_np, reward, done = (
                self.policy.train_step(
                    spiderbot_pose_msg,
                    delta_time
                )
            )
        self.locomotion_node.publish_step_reward(
            reward
        )

        if done:
            self.locomotion_node.get_logger().info('Training episode done')
            # Wait until a reset
            self.is_resetting = True

        return action_np

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
                    self.leg_names,
                    target_angles_per_leg
                )
        self.locomotion_node.publish_angles(msg)

    def set_training_target(self, set_training_target_msg):
        """Set the target and the estimated time to reach it."""
        super().set_training_target(set_training_target_msg)

    def reset(self):
        """Reset the neural network."""
        if self.population_training:
            episode_reward = self.population_trainer.get_episode_reward()
            epoch_reward = self.population_trainer.get_epoch_reward()
            self.population_trainer.start_new_training_episode()
        else:
            episode_reward = self.policy.get_episode_reward()
            self.policy.start_new_training_episode()

        # Save the weights periodically
        self.num_episodes += 1
        if self.num_episodes % self.episode_save_interval == 1:
            if self.population_training:
                self.population_trainer.save_population_checkpoint()
            else:
                self.policy.save_weights()

        if self.population_training:
            self.population_trainer.set_target(
                self.time_to_reach_target_s,
                self.target
            )
        else:
            self.policy.set_target(
                self.time_to_reach_target_s,
                self.target
            )

        self.locomotion_node.publish_episode_reward(
            episode_reward
        )
        if (self.population_training and epoch_reward):
            self.locomotion_node.publish_epoch_reward(
                epoch_reward
            )

        super().reset()

    def reset_learned_weights(self):
        """Back up the current weights and start with new random weights."""
        if self.population_training:
            self.population_trainer.reset_checkpoints()
        else:
            self.policy.reset_learned_weights()
