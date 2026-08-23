"""Runs a node for providing planning to a Spiderbot."""

import rclpy
from rclpy.executors import ExternalShutdownException

from .planning_node import PlanningNode


def main(args=None):
    """Run the Spiderbot planner."""
    rclpy.init(args=args)
    planning_node = None
    try:
        planning_node = PlanningNode()
        while rclpy.ok():
            rclpy.spin_once(planning_node, timeout_sec=0)
            pass
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        if planning_node is not None:
            planning_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
