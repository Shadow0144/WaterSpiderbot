"""Spiderbot locomotion node."""

from geometry_msgs.msg import Vector3

from rclpy.node import Node

from spiderbot_interfaces.msg import LegSetTargets

from std_msgs.msg import Float32

from . import MoveToPointLocomotionModule


class SpiderbotLocomotionNode(Node):
    """Spiderbot locomotion."""

    def __init__(self):
        """Initialize and run a Spiderbot locomotor."""
        super().__init__('locomotion_node')

        self.locomotion_module = MoveToPointLocomotionModule()

        self.leg_set_targets_publisher = self.create_publisher(
            LegSetTargets, 'set_leg_set_targets', 10)

        self.simulation_delta_time_subscription = self.create_subscription(
            Float32,
            'simulation_delta_time',
            self.simulation_delta_time_callback,
            10
        )
        self.simulation_delta_time_subscription

    def simulation_delta_time_callback(self, msg):
        """Publish a set of leg targets."""
        self.locomotion_module.walk_forward(msg.data)
        msg = LegSetTargets()
        msg.leg_l_i_target = (Vector3(
                x=self.locomotion_module.current_targets['l_i'][0],
                y=self.locomotion_module.current_targets['l_i'][1],
                z=self.locomotion_module.current_targets['l_i'][2],
            )
        )
        msg.leg_l_ii_target = (Vector3(
                x=self.locomotion_module.current_targets['l_ii'][0],
                y=self.locomotion_module.current_targets['l_ii'][1],
                z=self.locomotion_module.current_targets['l_ii'][2],
            )
        )
        msg.leg_l_iii_target = (Vector3(
                x=self.locomotion_module.current_targets['l_iii'][0],
                y=self.locomotion_module.current_targets['l_iii'][1],
                z=self.locomotion_module.current_targets['l_iii'][2],
            )
        )
        msg.leg_l_iv_target = (Vector3(
                x=self.locomotion_module.current_targets['l_iv'][0],
                y=self.locomotion_module.current_targets['l_iv'][1],
                z=self.locomotion_module.current_targets['l_iv'][2],
            )
        )
        msg.leg_r_i_target = (Vector3(
                x=self.locomotion_module.current_targets['r_i'][0],
                y=self.locomotion_module.current_targets['r_i'][1],
                z=self.locomotion_module.current_targets['r_i'][2],
            )
        )
        msg.leg_r_ii_target = (Vector3(
                x=self.locomotion_module.current_targets['r_ii'][0],
                y=self.locomotion_module.current_targets['r_ii'][1],
                z=self.locomotion_module.current_targets['r_ii'][2],
            )
        )
        msg.leg_r_iii_target = (Vector3(
                x=self.locomotion_module.current_targets['r_iii'][0],
                y=self.locomotion_module.current_targets['r_iii'][1],
                z=self.locomotion_module.current_targets['r_iii'][2],
            )
        )
        msg.leg_r_iv_target = (Vector3(
                x=self.locomotion_module.current_targets['r_iv'][0],
                y=self.locomotion_module.current_targets['r_iv'][1],
                z=self.locomotion_module.current_targets['r_iv'][2],
            )
        )
        self.leg_set_targets_publisher.publish(msg)
