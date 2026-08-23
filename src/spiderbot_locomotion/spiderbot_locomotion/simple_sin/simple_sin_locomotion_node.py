"""Spiderbot locomotion node using a simple sin function."""

from .simple_sin_module import SimpleSinModule
from ..locomotion_node import LocomotionNode


class SimpleSinLocomotionNode(LocomotionNode):
    """Spiderbot locomotion using a simple sin function."""

    def __init__(self):
        """Initialize and run a Spiderbot locomotor."""
        super().__init__('simple_sin_locomotion_node')

        # Set the module after getting the description
        self.locomotion_module = SimpleSinModule(
            self,
            self.spiderbot_description
        )
