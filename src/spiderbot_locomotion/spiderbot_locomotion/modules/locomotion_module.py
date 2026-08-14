"""Abstract class for creating locomotion modules."""

import spiderbot_utilities as utils


class LocomotionModule:
    """Abstract class for creating locomotion modules."""

    def __init__(self, locomotion_node, spiderbot_description):
        """Initialize the module."""
        self.locomotion_node = locomotion_node
        self.spiderbot_description = spiderbot_description
        self.leg_descriptions, self.leg_names, self.segment_lengths_per_leg = (
            utils.convert_spiderbot_description_to_lists(
                self.spiderbot_description
                )
        )
        self.last_timestamp = -1.0
        self.time_to_reach_target_s = 0
        self.target_x = 0
        self.target_y = 0
        self.target_theta = 0
        self.target = [
            self.target_x,
            self.target_y,
            self.target_theta,
        ]

        self.is_resetting = False

    def set_training_mode_enabled(self, training_mode_enabled):
        """Toggle if training mode is enabled."""
        self.training_mode_enabled = training_mode_enabled

    def update(self, spiderbot_pose_msg):
        """Walk the Spiderbot towards its target."""
        pass

    def set_training_target(self, set_training_target_msg):
        """Set a target (x, y + rotation) for the Spiderbot to move towards."""
        self.time_to_reach_target_s = (
            set_training_target_msg.time_to_reach_target_s
        )
        self.target_x = set_training_target_msg.target_x
        self.target_y = set_training_target_msg.target_y
        self.target_theta = set_training_target_msg.target_theta
        self.target = [
            self.target_x,
            self.target_y,
            self.target_theta,
        ]

    def get_delta_time_from_msg(self, spiderbot_pose_msg):
        """Get the change in time between messages."""
        if (self.last_timestamp < 0.0):
            # Skip the first update to make sure we have an
            # appropriate delta time
            self.last_timestamp = spiderbot_pose_msg.timestamp
            return -1
        else:
            delta_time = spiderbot_pose_msg.timestamp - self.last_timestamp
            self.last_timestamp = spiderbot_pose_msg.timestamp
            return delta_time

    def reset(self):
        """After finishing a reset, reset the is_resetting flag."""
        self.is_resetting = False

    def reset_learned_weights(self):
        """Backup the current weights and start with new random weights."""
        pass
