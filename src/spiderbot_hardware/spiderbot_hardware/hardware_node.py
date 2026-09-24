"""Provide an inferface for the hardware of a Spiderbot."""

from rclpy.node import Node


class HardwareNode(Node):
    """A hardware node for a Spiderbot."""

    def __init__(self):
        """Initialize the hardware node."""
        super().__init__('hardware_node')

    def is_running(self):
        """Return if the node is running."""
        return False
