"""Creates a Simulation node for a Spiderbot."""


import rclpy
from rclpy.executors import ExternalShutdownException

from .simulation_node import SimulationNode


def main(args=None):
    """Run the Spiderbot simulation."""
    rclpy.init(args=args)
    simulation_node = None
    try:
        simulation_node = SimulationNode()
        while rclpy.ok() and simulation_node.is_running():
            rclpy.spin_once(simulation_node, timeout_sec=0)
            simulation_node.update()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Exit on interrupt
    finally:
        if simulation_node is not None:
            simulation_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
