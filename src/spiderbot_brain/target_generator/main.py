"""Creates a Target Generator node for a Spiderbot."""


import rclpy
from rclpy.executors import ExternalShutdownException

from .target_generator_node import TargetGeneratorNode


def main(args=None):
    """Run the Spiderbot brain."""
    rclpy.init(args=args)
    target_generator_node = None
    try:
        target_generator_node = TargetGeneratorNode()
        while rclpy.ok() and target_generator_node.is_running():
            rclpy.spin_once(target_generator_node, timeout_sec=0)
            target_generator_node.update()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        if target_generator_node is not None:
            target_generator_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
