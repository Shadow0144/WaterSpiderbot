"""Runs a node for locomoting a Spiderbot."""

import rclpy
from rclpy.executors import ExternalShutdownException

from .kinematics_node import SpiderbotKinematicsNode


def main(args=None):
    """Run the Spiderbot kinematics calculator."""
    rclpy.init(args=args)
    kinematics_node = None
    try:
        kinematics_node = SpiderbotKinematicsNode()
        while rclpy.ok() and kinematics_node.is_running():
            rclpy.spin_once(kinematics_node, timeout_sec=0)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        if kinematics_node is not None:
            kinematics_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
