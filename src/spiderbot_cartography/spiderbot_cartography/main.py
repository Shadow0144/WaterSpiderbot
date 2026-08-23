"""Creates a Cartography node for a Spiderbot."""


import rclpy
from rclpy.executors import ExternalShutdownException

from .cartography_node import CartographyNode


def main(args=None):
    """Run the Spiderbot cartography."""
    rclpy.init(args=args)
    cartography_node = None
    try:
        cartography_node = CartographyNode()
        while rclpy.ok():
            rclpy.spin_once(cartography_node, timeout_sec=0)
            pass
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        if cartography_node is not None:
            cartography_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
