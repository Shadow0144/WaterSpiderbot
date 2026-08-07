"""A locomotion module using a Deep Neural Network."""

from .locomotion_module import LocomotionModule


class DNNModule(LocomotionModule):
    """A locomotion module using a Deep Neural Network."""

    def __init__(self, locomotion_node, spiderbot_description):
        """Initialize the locomotion module."""
        super().__init__(locomotion_node, spiderbot_description)
