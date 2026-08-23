"""Creates a Vision node for a Spiderbot."""


import rclpy
from rclpy.executors import ExternalShutdownException

from .vision_node import VisionNode


def main(args=None):
    """Run the Spiderbot vision node."""
    rclpy.init(args=args)
    vision_node = None
    try:
        vision_node = VisionNode()
        while rclpy.ok() and vision_node.is_running():
            rclpy.spin_once(vision_node, timeout_sec=0)
            pass
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        if vision_node is not None:
            vision_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
