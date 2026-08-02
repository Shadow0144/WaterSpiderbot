"""Simulate a Spiderbot and launch an interactive viewer."""

import time

import mujoco
import mujoco.viewer

from rclpy.node import Node

from spiderbot_description import Spiderbot

from spiderbot_interfaces.msg import LegSetTargets

from std_msgs.msg import Float32


def convert_vector3_to_list(vector3):
    """Convert a Vector3 object to a list."""
    return (vector3.x, vector3.y, vector3.z)


class SimulationNode(Node):
    """A simulation node for a Spiderbot."""

    def __init__(self):
        """Initialize and run a simulation."""
        super().__init__('simulation_node')

        self.last_timestamp = time.time()

        self.spider = Spiderbot()

        self.leg_set_targets_subscription = self.create_subscription(
            LegSetTargets,
            'set_leg_set_targets',
            self.set_leg_set_targets_callback,
            10
        )
        self.leg_set_targets_subscription

        self.delta_time_publisher = self.create_publisher(
            Float32,
            'simulation_delta_time',
            10
        )

        self.create_mujoco_viewer()

    def set_leg_set_targets_callback(self, msg):
        """Update the targets for every leg of the spiderbot."""
        targets = {
            'l_i': convert_vector3_to_list(msg.leg_l_i_target),
            'l_ii': convert_vector3_to_list(msg.leg_l_ii_target),
            'l_iii': convert_vector3_to_list(msg.leg_l_iii_target),
            'l_iv': convert_vector3_to_list(msg.leg_l_iv_target),
            'r_i': convert_vector3_to_list(msg.leg_r_i_target),
            'r_ii': convert_vector3_to_list(msg.leg_r_ii_target),
            'r_iii': convert_vector3_to_list(msg.leg_r_iii_target),
            'r_iv': convert_vector3_to_list(msg.leg_r_iv_target),
        }
        self.spider.set_claw_targets(targets)

    def create_mujoco_viewer(self):
        """Create the simulation viewer to test the Spiderbot."""
        self.viewer = mujoco.viewer.launch_passive(self.spider.model,
                                                   self.spider.data)
        self.viewer.cam.azimuth = 270
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 2.0
        self.viewer.cam.lookat[:] = [0, 0, 0.25]

        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    def update_viewer(self):
        """Update the simulation viewer."""
        if self.viewer.is_running():
            step_start = time.time()

            delta_time = step_start - self.last_timestamp
            self.last_timestamp = step_start
            delta_time_msg = Float32()
            delta_time_msg.data = delta_time
            self.delta_time_publisher.publish(delta_time_msg)

            mujoco.mj_step(self.spider.model, self.spider.data)

            self.viewer.sync()

            time_until_next_step = (
                self.spider.model.opt.timestep - (time.time() - step_start)
            )
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
