"""Abstract class for creating locomotion modules."""


class LocomotionModule:
    """Abstract class for creating locomotion modules."""

    def __init__(self, locomotion_node):
        """Initialize the module."""
        self.locomotion_node = locomotion_node
        self.leg_names = self.locomotion_node.leg_names
        self.segment_lengths = self.locomotion_node.segment_lengths

    def walk_forward(self, time, leg_set):
        """Walk the Spiderbot forward."""
        pass
