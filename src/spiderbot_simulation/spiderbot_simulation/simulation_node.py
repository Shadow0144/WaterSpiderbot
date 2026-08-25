"""Simulate a Spiderbot and launch an interactive viewer."""

import time

import mujoco

import numpy as np

import rclpy
from rclpy.node import Node

from spiderbot_interfaces.msg import LegTargets
from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.msg import SpiderbotTargetPose
from spiderbot_interfaces.msg import TrainingTarget
from spiderbot_interfaces.srv import GetSpiderbotDescription

import spiderbot_utilities as utils

from std_msgs.msg import Float64

from std_srvs.srv import Empty

from .simulation_viewer import SimulationViewer


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
        self.spiderbot_description = self._request_spiderbot_description()
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

        target_publish_rate_ps = 60.0
        self.publish_interval = 1.0 / target_publish_rate_ps
        self.last_timestamp = time.time()

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

        self.step_reward_subscription = self.create_subscription(
            Float64,
            'step_reward',
            self.step_reward_callback,
            10
        )

        self.episode_reward_subscription = self.create_subscription(
            Float64,
            'episode_reward',
            self.episode_reward_callback,
            10
        )

        self.epoch_reward_subscription = self.create_subscription(
            Float64,
            'epoch_reward',
            self.epoch_reward_callback,
            10
        )

        self.reset_simulation_service = self.create_service(
            Empty,
            'reset_simulation',
            self.reset_simulation_callback
        )

        self.last_timestamp = time.time()

        self.viewer = SimulationViewer(self.model, self.data)

        self.get_logger().info('Spiderbot simulation node started')

    def destroy_node(self):
        """Destroy the window and finish destroying the node."""
        self.viewer.destroy()
        return super().destroy_node()

    def is_running(self):
        """Return if the node is running or if it's ready to shut down."""
        return self.viewer.is_running()

    def _request_spiderbot_description(self):
        """Get the spec xml from the description."""
        request = GetSpiderbotDescription.Request()
        future = self.spiderbot_description_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

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

    def step_reward_callback(self, msg):
        """Enable displaying the step reward and update the reward value."""
        self.viewer.update_step_reward(msg.data)

    def episode_reward_callback(self, msg):
        """Enable displaying the episode reward and update the reward value."""
        self.viewer.update_episode_reward(msg.data)

    def epoch_reward_callback(self, msg):
        """Enable displaying the epoch reward and update the reward value."""
        self.viewer.update_epoch_reward(msg.data)

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

    def _publish_pose(self, current_timestamp):
        """Publish the current pose."""
        spiderbot_pose_msg = utils.construct_pose_msg(
            self.last_timestamp,
            self.body,
            self.leg_names,
            self.legs
        )
        self.spiderbot_pose_publisher.publish(spiderbot_pose_msg)

    def update(self):
        """Step the physics, publish the pose, and update the render."""
        if not self.viewer.is_running():
            return

        current_timestamp = time.time()

        # Step the physics
        mujoco.mj_step(self.model, self.data)

        # Publish the current pose if enough time has elapsed
        if current_timestamp - self.last_timestamp >= self.publish_interval:
            self.last_timestamp = current_timestamp
            self._publish_pose(current_timestamp)

        # Update the renderer
        self.viewer.update(current_timestamp)

        # Sleep until it is time for the next simulation step
        time_elapsed = time.time() - current_timestamp
        time_until_next_step = (
            self.model.opt.timestep - time_elapsed
        )
        if time_until_next_step > 0.0:
            time.sleep(time_until_next_step)
