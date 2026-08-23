"""Perform SLAM."""

from rclpy.node import Node


class CartographyNode(Node):
    """A cartography node for a Spiderbot."""

    def __init__(self):
        """Initialize and run a cartographer."""
        super().__init__('cartography_node')
