"""Runs a node for locomoting a Spiderbot."""

import rclpy
from rclpy.executors import ExternalShutdownException

from .locomotion_node import SpiderbotLocomotionNode


def main(args=None):
    """Run the Spiderbot locomotor."""
    rclpy.init(args=args)
    locomotion_node = SpiderbotLocomotionNode()
    try:
        rclpy.spin(locomotion_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        locomotion_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
