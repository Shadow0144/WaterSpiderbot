"""Provide a teleop interface for a Spiderbot."""

from rclpy.node import Node


class TeleopNode(Node):
    """A teleop node for a Spiderbot."""

    def __init__(self):
        """Initialize the teleop node."""
        super().__init__('teleop_node')

    def is_running(self):
        """Return if the node is running."""
        return False
