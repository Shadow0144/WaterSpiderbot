"""Abstract class for creating locomotion modules."""

import spiderbot_utilities as utils


class LocomotionModule:
    """Abstract class for creating locomotion modules."""

    def __init__(self, locomotion_node, spiderbot_description):
        """Initialize the module."""
        self.locomotion_node = locomotion_node
        self.spiderbot_description = spiderbot_description
        (
            self.leg_descriptions,
            self.leg_names,
            self.segment_lengths_per_leg,
            self.spec,
            self.model,
            self.data,
            self.body,
            self.legs
        ) = utils.convert_spiderbot_description_to_variables(
            self.spiderbot_description
        )

        self.last_timestamp = -1.0

        self.target = None

        self.is_resetting = False

    def set_training_mode_enabled(self, training_mode_enabled):
        """Toggle if training mode is enabled."""
        self.training_mode_enabled = training_mode_enabled

    def _update_delta_time(self, spiderbot_pose_msg):
        """Get the change in time between messages."""
        timestamp = spiderbot_pose_msg.timestamp
        if self.last_timestamp < 0.0:
            self.last_timestamp = timestamp
            self.delta_time = 0.0
        else:
            self.delta_time = timestamp - self.last_timestamp
            self.last_timestamp = timestamp

    def update(self, spiderbot_pose_msg):
        """Walk the Spiderbot towards its target."""
        self._update_delta_time(spiderbot_pose_msg)

    def set_training_target(self, set_training_target_msg):
        """Set a target (x, y + rotation) for the Spiderbot to move towards."""
        self.target_x = set_training_target_msg.target_x
        self.target_y = set_training_target_msg.target_y
        self.target_theta = set_training_target_msg.target_theta
        self.target = [
            self.target_x,
            self.target_y,
            self.target_theta
        ]

    def start_training_episode(self):
        """Start a new training episode."""
        pass

    def reset(self):
        """After finishing a reset, reset the is_resetting flag."""
        self.is_resetting = False

    def reset_learned_weights(self):
        """Backup the current weights and start with new random weights."""
        pass
