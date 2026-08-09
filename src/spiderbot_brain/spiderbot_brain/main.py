"""Creates a Brain node for a Spiderbot."""


import rclpy
from rclpy.executors import ExternalShutdownException

from .brain_node import BrainNode


def main(args=None):
    """Run the Spiderbot brain."""
    rclpy.init(args=args)
    brain_node = None
    try:
        brain_node = BrainNode()
        while rclpy.ok():
            if brain_node.training_mode_enabled:
                brain_node.perform_training_step()
            rclpy.spin_once(brain_node, timeout_sec=0)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        if brain_node is not None:
            brain_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
