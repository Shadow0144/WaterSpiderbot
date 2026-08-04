"""Runs a node for locomoting a Spiderbot."""

import time

import rclpy
from rclpy.executors import ExternalShutdownException

from .description_node import DescriptionNode


def main(args=None):
    """Run the Spiderbot descriptor."""
    rclpy.init(args=args)
    description_node = DescriptionNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(description_node, timeout_sec=0)
            time.sleep(0.3)  # Seconds
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        description_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
