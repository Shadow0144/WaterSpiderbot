"""Creates a Teleop node for a Spiderbot."""


import rclpy
from rclpy.executors import ExternalShutdownException

from .teleop_node import TeleopNode


def main(args=None):
    """Run the Spiderbot brain."""
    rclpy.init(args=args)
    teleop_node = None
    try:
        teleop_node = TeleopNode()
        while rclpy.ok():
            rclpy.spin_once(teleop_node, timeout_sec=0)
            teleop_node.update()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        if teleop_node is not None:
            teleop_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
