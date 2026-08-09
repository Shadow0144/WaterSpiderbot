"""Spiderbot locomotion node."""

from rcl_interfaces.msg import SetParametersResult

import rclpy
from rclpy.node import Node

from spiderbot_interfaces.msg import LegTargets
from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.msg import SpiderbotTargetPose
from spiderbot_interfaces.msg import TrainingTarget
from spiderbot_interfaces.srv import GetSpiderbotDescription

from std_msgs.msg import Float64

from std_srvs.srv import Empty
from std_srvs.srv import SetBool
from std_srvs.srv import Trigger

from .modules import DNNModule
from .modules import HandcraftedAngleModule
from .modules import HandcraftedPointModule
from .modules import SimpleSinModule


class SpiderbotLocomotionNode(Node):
    """Spiderbot locomotion."""

    def __init__(self):
        """Initialize and run a Spiderbot locomotor."""
        super().__init__('locomotion_node')

        self.get_logger().info('Starting spiderbot locomotion node')

        self.declare_parameter('locomotion_module',
                               'dnn')
        self.locomotion_module_type = (
            self.get_parameter('locomotion_module').value
        )

        self.declare_parameter('training_mode_enabled',
                               True)
        self.training_mode_enabled = (
            self.get_parameter('training_mode_enabled').value
        )

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

        # Set the module after getting the description
        self.set_locomotion_module()

        self.add_on_set_parameters_callback(self.parameter_changed_callback)

        self.spiderbot_target_pose_publisher = self.create_publisher(
            SpiderbotTargetPose, 'spiderbot_target_pose', 10)

        self.leg_set_targets_publisher = self.create_publisher(
            LegTargets, 'set_leg_targets', 10)

        self.current_step_reward_publisher = self.create_publisher(
            Float64, 'current_step_reward', 10)

        self.training_run_reward_publisher = self.create_publisher(
            Float64, 'training_run_reward', 10)

        self.spiderbot_pose_subscription = self.create_subscription(
            SpiderbotPose,
            'spiderbot_pose',
            self.spiderbot_pose_callback,
            10
        )

        self.reset_simulation_client = self.create_client(
            Empty,
            'reset_simulation'
        )

        self.training_target_subscription = self.create_subscription(
            TrainingTarget,
            'training_target',
            self.training_target_callback,
            10
        )

        self.set_training_mode_enabled_service = self.create_service(
            SetBool,
            'set_training_mode_enabled',
            self.set_training_mode_enabled_callback
        )

        self.reset_learned_weights_service = self.create_service(
            Trigger,
            'reset_learned_weights',
            self.reset_learned_weights_callback
        )

        self.simulation_reset_queued = False

        self.get_logger().info('Spiderbot locomotion node started')

    def is_running(self):
        """Return if the node is running or if it's ready to shut down."""
        return True

    def parameter_changed_callback(self, params):
        """React to parameters updating."""
        for param in params:
            if param.name == 'locomotion_module':
                self.locomotion_module_type = param.value
                self.set_locomotion_module()
            elif param.name == 'training_mode_enabled':
                if self.locomotion_module is not None:
                    self.locomotion_module.set_training_mode_enabled(
                        param.value
                    )
        return SetParametersResult(successful=True)

    def set_locomotion_module(self):
        """Set the locomotion module."""
        if self.locomotion_module_type == 'simple_sin':
            self.locomotion_module = SimpleSinModule(
                self,
                self.spiderbot_description)
        elif self.locomotion_module_type == 'handcrafted_angle':
            self.locomotion_module = HandcraftedAngleModule(
                self,
                self.spiderbot_description)
        elif self.locomotion_module_type == 'handcrafted_point':
            self.locomotion_module = HandcraftedPointModule(
                self,
                self.spiderbot_description)
        elif self.locomotion_module_type == 'dnn':
            self.locomotion_module = DNNModule(
                self,
                self.spiderbot_description)

    def request_spiderbot_description(self):
        """Get the spec xml from the description."""
        request = GetSpiderbotDescription.Request()
        future = self.spiderbot_description_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def spiderbot_pose_callback(self, msg):
        """Publish a set of leg targets whenever a new pose is received."""
        if self.locomotion_module is not None:
            self.locomotion_module.update(msg)

    def target_location_callback(self, msg):
        """Set the target for the locomotion module to approach."""
        if self.locomotion_module is not None:
            self.locomotion_module.set_target(msg)

    def publish_angles(self, msg):
        """Publish target angles for the leg actuators."""
        self.spiderbot_target_pose_publisher.publish(msg)

    def publish_points(self, msg):
        """Publish target points for the leg to reach for."""
        self.leg_set_targets_publisher.publish(msg)

    def publish_current_step_reward(self, reward):
        """Publish the reward for the current step."""
        msg = Float64()
        msg.data = reward
        self.current_step_reward_publisher.publish(msg)

    def publish_training_run_reward(self, reward):
        """Publish the reward for the full training run."""
        msg = Float64()
        msg.data = reward
        self.training_run_reward_publisher.publish(msg)

    def training_target_callback(self, msg):
        """Reset the simuation and has the Spiderbot move to the target."""
        if self.locomotion_module is not None:
            self.locomotion_module.set_training_target(msg)
            self.queue_simulation_reset()

    def set_training_mode_enabled_callback(self, request, response):
        """Toggle if training mode is enabled."""
        self.locomotion_module.set_training_mode_enabled(request.data)
        response.success = True
        response.message = 'Success'
        return response

    def reset_learned_weights_callback(self, request, response):
        """Backup the current weights and start with new random weights."""
        self.locomotion_module.reset_learned_weights()
        response.success = True
        response.message = 'Success'
        return response

    def queue_simulation_reset(self):
        """Queue a simulation reset for the next available chance."""
        """(Solves issues with threads)"""
        self.simulation_reset_queued = True

    def reset_simulation(self):
        """Request the simulation to reset."""
        request = Empty.Request()
        future = self.reset_simulation_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        self.simulation_reset_queued = False
        self.locomotion_module.reset()
