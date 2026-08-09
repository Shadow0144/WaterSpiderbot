"""A locomotion module using a Deep Neural Network."""

import time

import spiderbot_utilities as utils

from .locomotion_module import LocomotionModule
from ..neural_network.locomotion_neural_network import LocomotionNeuralNetwork


class DNNModule(LocomotionModule):
    """A locomotion module using a Deep Neural Network."""

    def __init__(self, locomotion_node, spiderbot_description):
        """Initialize the locomotion module."""
        super().__init__(locomotion_node, spiderbot_description)

        self.nn = LocomotionNeuralNetwork()

        self.target = None

        self.training = True
        self.num_intervals = 0
        self.save_interval = 10
        if self.nn.get_model_weights_exist():
            self.nn.load_weights()

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
            angles = self.nn.forward(
                spiderbot_pose_msg
            )
        else:
            angles = self.training_step(
                spiderbot_pose_msg,
                delta_time
            )
        if angles is not None:
            self.publish_angles(angles)

    def training_step(self, spiderbot_pose_msg, delta_time):
        """Handle a training step."""
        action_np, reward, done = (
            self.nn.train_step(spiderbot_pose_msg,
                               delta_time)
        )
        self.locomotion_node.publish_current_step_reward(
            reward
        )

        if done:
            # Save the weights periodically
            self.num_intervals += 1
            if self.num_intervals % self.save_interval == 0:
                self.nn.save_weights()

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
        self.nn.set_target(self.time_to_reach_goal_s, self.target)

    def reset(self):
        """Reset the neural network."""
        self.locomotion_node.publish_training_run_reward(
            self.nn.training_run_reward /
            self.nn.time_to_goal_s
        )
        super().reset()
        self.nn.reset()

    def reset_learned_weights(self):
        """Backup the current weights and start with new random weights."""
        self.nn.reset_learned_weights()
