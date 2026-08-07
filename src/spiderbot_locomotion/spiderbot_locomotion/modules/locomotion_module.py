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

    def set_training_mode_enabled(self, training_mode_enabled):
        """Toggle if training mode is enabled."""
        self.training_mode_enabled = training_mode_enabled

    def update(self, spiderbot_pose_msg):
        """Walk the Spiderbot forward."""
        pass
