"""Runs a node for locomoting a Spiderbot."""

import rclpy
from rclpy.executors import ExternalShutdownException

from .locomotion_node import SpiderbotLocomotionNode


def main(args=None):
    """Run the Spiderbot locomotor."""
    rclpy.init(args=args)
    locomotion_node = None
    try:
        locomotion_node = SpiderbotLocomotionNode()
        while rclpy.ok() and locomotion_node.is_running():
            rclpy.spin_once(locomotion_node, timeout_sec=0)
            if locomotion_node.simulation_reset_queued:
                locomotion_node.reset_simulation()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        if locomotion_node is not None:
            locomotion_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
