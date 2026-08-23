"""Handle vision logic for a Spiderbot."""

from rclpy.node import Node


class VisionNode(Node):
    """A vision node for a Spiderbot."""

    def __init__(self):
        """Initialize and run the vision node."""
        super().__init__('vision_node')
