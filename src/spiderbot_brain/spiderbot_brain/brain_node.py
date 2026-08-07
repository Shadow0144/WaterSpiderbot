"""Perform high-level planning and training."""

from rclpy.node import Node


class BrainNode(Node):
    """A brain node for a Spiderbot."""

    def __init__(self):
        """Initialize and run a brain."""
        super().__init__('brain_node')

        self.declare_parameter('training_mode_enabled',
                               True)
        self.training_mode_enabled = (
            self.get_parameter('training_mode_enabled').value
        )
