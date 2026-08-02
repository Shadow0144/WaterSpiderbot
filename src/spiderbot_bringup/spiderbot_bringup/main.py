"""Runs the main Spiderbot test."""

import rclpy
from rclpy.node import Node

import spiderbot_simulation


class SpiderbotSimulationNode(Node):
    """Spiderbot simulation."""

    def __init__(self):
        """Initialize and run a Spiderbot simuation."""
        super().__init__('spiderbot_simulation')
        spiderbot_simulation.run_spiderbot_test()


def main(args=None):
    """Run the Spiderbot simulation."""
    rclpy.init(args=args)
    node = SpiderbotSimulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # Exit on interrupt
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
