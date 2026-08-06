"""A locomotion module using a Technical Neural Network."""

from .locomotion_module import LocomotionModule


class TNNModule(LocomotionModule):
    """A locomotion module using a Technical Neural Network."""

    def __init__(self, locomotion_node, spiderbot_description):
        """Initialize the locomotion module."""
        super().__init__(locomotion_node, spiderbot_description)
