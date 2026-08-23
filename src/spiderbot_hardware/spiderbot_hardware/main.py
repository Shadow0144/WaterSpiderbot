"""Runs a node for interfacing with the hardware of a Spiderbot."""

import rclpy
from rclpy.executors import ExternalShutdownException

from .hardware_node import HardwareNode


def main(args=None):
    """Run the Spiderbot hardware interface."""
    rclpy.init(args=args)
    hardware_node = None
    try:
        hardware_node = HardwareNode()
        while rclpy.ok() and hardware_node.is_running():
            rclpy.spin_once(hardware_node, timeout_sec=0)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        if hardware_node is not None:
            hardware_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
