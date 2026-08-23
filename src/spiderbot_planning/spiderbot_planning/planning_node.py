"""Spiderbot planning node."""

from rclpy.node import Node


class PlanningNode(Node):
    """Spiderbot planning."""

    def __init__(self):
        """Initialize and run a Spiderbot locomotor."""
        super().__init__('planning_node')
