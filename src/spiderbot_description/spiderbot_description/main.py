"""Runs a node for describing a Spiderbot."""

import rclpy
from rclpy.executors import ExternalShutdownException

from .description_node import DescriptionNode


def main(args=None):
    """Run the Spiderbot descriptor."""
    rclpy.init(args=args)
    description_node = None
    try:
        description_node = DescriptionNode()
        while rclpy.ok() and description_node.is_running():
            rclpy.spin_once(description_node, timeout_sec=0)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        if description_node is not None:
            description_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
