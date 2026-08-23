"""Spiderbot locomotion node using hand-crafted point targets."""

from .handcrafted_points_module import HandcraftedPointsModule
from ..locomotion_node import LocomotionNode


class HandCraftedPointsLocomotionNode(LocomotionNode):
    """Spiderbot locomotion using hand-crafted point targets."""

    def __init__(self):
        """Initialize and run a Spiderbot locomotor."""
        super().__init__('handcrafted_points_locomotion_node')

        # Set the module after getting the description
        self.locomotion_module = HandcraftedPointsModule(
            self,
            self.spiderbot_description
        )
