"""Creates a Simulation node for a Spiderbot."""


import rclpy
from rclpy.executors import ExternalShutdownException

from .simulation_node import SimulationNode


def main(args=None):
    """Run the Spiderbot simulation."""
    rclpy.init(args=args)
    simulation_node = SimulationNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(simulation_node, timeout_sec=0)
            if simulation_node.viewer.is_running():
                simulation_node.update_viewer()
            else:
                break  # Viewer has closed, shut down
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        simulation_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
