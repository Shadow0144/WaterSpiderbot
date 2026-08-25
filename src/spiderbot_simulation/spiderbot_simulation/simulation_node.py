"""Simulate a Spiderbot and launch an interactive viewer."""

import time

import mujoco
import mujoco.viewer

import numpy as np

import rclpy
from rclpy.node import Node

from spiderbot_interfaces.msg import LegTargets
from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.msg import SpiderbotTargetPose
from spiderbot_interfaces.msg import TrainingTarget
from spiderbot_interfaces.srv import GetSpiderbotDescription

import spiderbot_utilities as utils

from std_srvs.srv import Empty


class SimulationNode(Node):
    """A simulation node for a Spiderbot."""

    def __init__(self):
        """Initialize and run a simulation."""
        super().__init__('simulation_node')

        self.get_logger().info('Starting spiderbot simulation node')

        self.spiderbot_description_client = self.create_client(
            GetSpiderbotDescription,
            'get_spiderbot_description')
        while not self.spiderbot_description_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                'Waiting on get_spec_xml service',
                once=True)
        self.spiderbot_description = self.request_spiderbot_description()
        self.get_logger().info('Spiderbot description received')

        (
            self.leg_descriptions,
            self.leg_names,
            self.segment_lengths_per_leg,
            self.spec,
            self.model,
            self.data,
            self.body,
            self.legs
        ) = utils.convert_spiderbot_description_to_variables(
            self.spiderbot_description
        )

        self.training_target_visible = False
        self.training_target_position = None
        self.training_target_quaternion = None
        self.training_target_z = 0.5
        self.training_target = self.data.body('training_target')
        training_target_body_id = self.model.body('training_target').id
        self.training_target_mocap_id = self.model.body_mocapid[
            training_target_body_id
        ]
        self.training_target_forward_geom_id = self.model.geom(
            'training_target_forward_geom'
        ).id
        self.training_target_backward_geom_id = self.model.geom(
            'training_target_backward_geom'
        ).id

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

        self.spiderbot_target_pose_subscription = self.create_subscription(
            SpiderbotTargetPose,
            'spiderbot_target_pose',
            self.spiderbot_target_pose_callback,
            10
        )

        self.training_target_subscription = self.create_subscription(
            TrainingTarget,
            'training_target',
            self.training_target_callback,
            10
        )

        self.reset_simulation_service = self.create_service(
            Empty,
            'reset_simulation',
            self.reset_simulation_callback
        )

        self.last_timestamp = time.time()

        self._create_mujoco_viewer()

        self.get_logger().info('Spiderbot simulation node started')

    def is_running(self):
        """Return if the node is running or if it's ready to shut down."""
        return self.viewer is not None and self.viewer.is_running()

    def request_spiderbot_description(self):
        """Get the spec xml from the description."""
        request = GetSpiderbotDescription.Request()
        future = self.spiderbot_description_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def _create_mujoco_viewer(self):
        """Create the simulation viewer to test the Spiderbot."""
        self.viewer = mujoco.viewer.launch_passive(self.model,
                                                   self.data)
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 2.0
        self.viewer.cam.lookat[:] = [0, 0, 0.25]

        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    def set_leg_targets_callback(self, msg):
        """Move the mocaps to the targets."""
        leg_target_values = msg.leg_targets
        if len(leg_target_values) != len(self.leg_names):
            return  # Break early
        leg_targets = dict(zip(self.leg_names, leg_target_values))
        for leg_name in self.leg_names:
            # Move the mocap to the target point
            self.legs[leg_name].set_mocap_target_visible(True)
            self.legs[leg_name].set_mocap_target(
                utils.convert_vector3_to_list(leg_targets[leg_name]))

    def spiderbot_target_pose_callback(self, msg):
        """Update the targets for the actuators."""
        leg_poses_values = msg.leg_poses
        if len(leg_poses_values) != len(self.leg_names):
            return  # Break early
        leg_poses = dict(zip(self.leg_names, leg_poses_values))
        for leg_name in self.leg_names:
            leg_pose = leg_poses[leg_name]
            self.legs[leg_name].set_target_qposes(
                leg_pose.coxa_qpos,
                leg_pose.femur_qpos,
                leg_pose.tibia_qpos)

    def reset_simulation_callback(self, request, response):
        """Reset simulation."""
        mujoco.mj_resetData(self.model, self.data)
        # Move the target back to where it should be if necessary
        if self.training_target_visible:
            self.data.mocap_pos[self.training_target_mocap_id] = (
                self.training_target_position
            )
            self.data.mocap_quat[self.training_target_mocap_id] = (
                self.training_target_quaternion
            )
        mujoco.mj_forward(self.model, self.data)
        for leg_name in self.leg_names:
            self.legs[leg_name].reset_leg()
        if self.viewer.is_running():
            self.viewer.sync()
        return response

    def training_target_callback(self, msg):
        """Move the target to the location and make it visible."""
        self.training_target_visible = True
        alpha = 0.75
        self.training_target_position = [
            msg.target_x,
            msg.target_y,
            self.training_target_z
        ]
        half_theta = msg.target_theta / 2.0
        self.training_target_quaternion = [
            np.cos(half_theta), 0.0, 0.0, np.sin(half_theta)
        ]
        self.model.geom_rgba[self.training_target_forward_geom_id, 3] = alpha
        self.model.geom_rgba[self.training_target_backward_geom_id, 3] = alpha
        self.data.mocap_pos[self.training_target_mocap_id] = (
                    self.training_target_position
                )
        self.data.mocap_quat[self.training_target_mocap_id] = (
            self.training_target_quaternion
        )

    def update_viewer(self):
        """Update the simulation viewer."""
        if self.viewer.is_running():
            step_start = time.time()

            self.last_timestamp = step_start
            spiderbot_pose_msg = utils.construct_pose_msg(
                self.last_timestamp,
                self.body,
                self.leg_names,
                self.legs
            )
            self.spiderbot_pose_publisher.publish(spiderbot_pose_msg)

            mujoco.mj_step(self.model, self.data)

            self.viewer.sync()

            time_until_next_step = (
                self.model.opt.timestep - (time.time() - step_start)
            )
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
