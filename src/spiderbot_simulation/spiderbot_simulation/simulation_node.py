"""Simulate a Spiderbot and launch an interactive viewer."""

import time

import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState

from spiderbot_interfaces.msg import LegPose
from spiderbot_interfaces.msg import LegTargets
from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.srv import GetSpiderbotDescription

from .spider_leg import SpiderLeg


def convert_vector3_to_list(vector3):
    """Convert a Vector3 object to a list."""
    return (vector3.x, vector3.y, vector3.z)


class SimulationNode(Node):
    """A simulation node for a Spiderbot."""

    leg_names = ['l_i', 'l_ii', 'l_iii', 'l_iv',
                 'r_i', 'r_ii', 'r_iii', 'r_iv']

    def __init__(self):
        """Initialize and run a simulation."""
        super().__init__('simulation_node')

        self.last_timestamp = time.time()

        self.spiderbot_description_client = self.create_client(
            GetSpiderbotDescription,
            'get_spiderbot_description')
        while not self.spiderbot_description_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info('Waiting on get_spec_xml service')
        spiderbot_description = self.request_spiderbot_description()

        self.leg_names = spiderbot_description.leg_names
        self.segment_lengths = dict(zip(self.leg_names,
                                        spiderbot_description.segment_lengths))
        self.spec = mujoco.MjSpec.from_string(
            spiderbot_description.spec_xml
        )
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        body_joint_id = self.model.joint('cephalothorax_joint').id
        self.body_joint_qpos_adr = self.model.jnt_qposadr[body_joint_id]
        self.legs = {}
        for leg_name in self.leg_names:
            self.legs[leg_name] = SpiderLeg(leg_name,
                                            self.segment_lengths[leg_name],
                                            self.model,
                                            self.data)

        self.spiderbot_pose_publisher = self.create_publisher(
            SpiderbotPose,
            'spiderbot_pose',
            10
        )

        self.set_leg_targets_subscription = self.create_subscription(
            LegTargets,
            'set_leg_targets',
            self.set_leg_targets_callback,
            10
        )
        self.set_leg_targets_subscription

        self.spiderbot_target_pose_subscription = self.create_subscription(
            SpiderbotPose,
            'spiderbot_target_pose',
            self.spiderbot_target_pose_callback,
            10
        )
        self.spiderbot_target_pose_subscription

        self.create_mujoco_viewer()

    def request_spiderbot_description(self):
        """Get the spec xml from the description."""
        request = GetSpiderbotDescription.Request()
        future = self.spiderbot_description_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def create_mujoco_viewer(self):
        """Create the simulation viewer to test the Spiderbot."""
        self.viewer = mujoco.viewer.launch_passive(self.model,
                                                   self.data)
        self.viewer.cam.azimuth = 270
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 2.0
        self.viewer.cam.lookat[:] = [0, 0, 0.25]

        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    def convert_vector3_to_list(self, vector3):
        """Convert geometry_msg Vector3 to a Python list."""
        return [vector3.x, vector3.y, vector3.z]

    def set_leg_targets_callback(self, msg):
        """Move the mocaps to the targets."""
        leg_target_values = msg.leg_targets
        leg_targets = dict(zip(self.leg_names, leg_target_values))
        for leg_name in self.leg_names:
            # Move the mocap to the target point
            self.legs[leg_name].set_mocap_target(
                self.convert_vector3_to_list(leg_targets[leg_name]))

    def spiderbot_target_pose_callback(self, msg):
        """Update the targets for the actuators."""
        leg_poses_values = msg.leg_poses
        leg_poses = dict(zip(self.leg_names, leg_poses_values))
        for leg_name in self.leg_names:
            leg_pose = leg_poses[leg_name]
            self.legs[leg_name].set_target_qposes(
                leg_pose.coxa_qpos,
                leg_pose.femur_qpos,
                leg_pose.tibia_qpos)

    def create_joint_state(self, name, qpose):
        """Construct a JointStateObject."""
        joint_state = JointState()
        joint_state.name = name
        joint_state.position = qpose
        return joint_state

    def construct_pose_msg(self):
        """Construct a pose message."""
        msg = SpiderbotPose()
        msg.timestamp = self.last_timestamp
        msg.body_joint_state = self.create_joint_state(
            'cephalothorax_joint',
            self.data.qpos[
                self.body_joint_qpos_adr:(self.body_joint_qpos_adr + 7)])
        leg_qposes = []
        for leg_name in self.leg_names:
            qposes = self.legs[leg_name].get_qposes()
            leg_pose = LegPose()
            leg_pose.leg_name = leg_name
            leg_pose.coxa_qpos = qposes[0]
            leg_pose.femur_qpos = qposes[1]
            leg_pose.tibia_qpos = qposes[2]
            leg_qposes.append(leg_pose)
        return msg

    def update_viewer(self):
        """Update the simulation viewer."""
        if self.viewer.is_running():
            step_start = time.time()

            self.last_timestamp = step_start
            spiderbot_pose_msg = self.construct_pose_msg()
            self.spiderbot_pose_publisher.publish(spiderbot_pose_msg)

            mujoco.mj_step(self.model, self.data)

            self.viewer.sync()

            time_until_next_step = (
                self.model.opt.timestep - (time.time() - step_start)
            )
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
