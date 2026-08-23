"""Spiderbot locomotion node using hand-crafted angle targets."""

from .handcrafted_angles_module import HandcraftedAnglesModule
from ..locomotion_node import LocomotionNode


class HandCraftedAnglesLocomotionNode(LocomotionNode):
    """Spiderbot locomotion using hand-crafted angle targets."""

    def __init__(self):
        """Initialize and run a Spiderbot locomotor."""
        super().__init__('handcrafted_angles_locomotion_node')

        # Set the module after getting the description
        self.locomotion_module = HandcraftedAnglesModule(
            self,
            self.spiderbot_description
        )
