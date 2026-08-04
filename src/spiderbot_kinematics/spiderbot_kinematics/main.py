"""Runs a node for locomoting a Spiderbot."""

import rclpy
from rclpy.executors import ExternalShutdownException

from .kinematics_node import SpiderbotKinematicsNode


def main(args=None):
    """Run the Spiderbot kinematics calculator."""
    rclpy.init(args=args)
    kinematics_node = SpiderbotKinematicsNode()
    try:
        rclpy.spin(kinematics_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        kinematics_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
