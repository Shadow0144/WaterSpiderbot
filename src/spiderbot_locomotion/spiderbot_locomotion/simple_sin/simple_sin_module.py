"""A locomotion module using a simple sin wave with alternating leg groups."""

import math
import time

import numpy as np

import spiderbot_utilities as utils

from ..locomotion_module import LocomotionModule


class SimpleSinModule(LocomotionModule):
    """A locomotion module using a simple sin wave."""

    def __init__(self, locomotion_node,
                 spiderbot_description,
                 phase_shift=(2.0 * math.pi / 3.0)):
        """Initialize the locomotion module."""
        super().__init__(locomotion_node, spiderbot_description)
        self.current_targets = {}
        self.time_elapsed = 0

        self.offset_phase = phase_shift

        self.group_1 = ['l_i', 'l_iii', 'r_ii', 'r_iv']
        self.group_2 = ['l_ii', 'l_iv', 'r_i', 'r_iii']

    def _walk_forward(self, delta_time):
        """Walk the Spiderbot forward."""
        self.time_elapsed += delta_time
        cos_phase = np.cos(self.time_elapsed)
        sin_phase = np.sin(self.time_elapsed)
        cos_offset_phase = np.cos(self.time_elapsed + self.offset_phase)
        sin_offset_phase = np.sin(self.time_elapsed + self.offset_phase)

        coxa_target_angle = math.radians(-30)
        femur_target_angle = math.radians(45)
        tibia_target_angle = math.radians(5)

        for leg_name in self.group_1:
            self.current_targets[leg_name] = [
                coxa_target_angle * cos_phase,
                femur_target_angle * cos_offset_phase,
                tibia_target_angle * cos_offset_phase
            ]
        for leg_name in self.group_2:
            self.current_targets[leg_name] = [
                coxa_target_angle * sin_phase,
                femur_target_angle * sin_offset_phase,
                tibia_target_angle * sin_offset_phase
            ]

        self.publish_angles()

    def update(self, spiderbot_pose_msg):
        """Walk the spiderbot forward."""
        delta_time = self.get_delta_time_from_msg(spiderbot_pose_msg)
        if delta_time > 0.0:
            self._walk_forward(delta_time)

    def publish_angles(self):
        """Publish target angles for the leg actuators."""
        msg = utils.construct_target_pose_msg(
                    time.time(),
                    self.leg_names,
                    self.current_targets
                )
        self.locomotion_node.publish_angles(msg)
