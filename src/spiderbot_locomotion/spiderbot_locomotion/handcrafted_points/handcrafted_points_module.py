"""A locomotion module using target points with alternating leg groups."""

import time
from enum import Enum, auto

import mujoco

import numpy as np

import spiderbot_utilities as utils

from ..locomotion_module import LocomotionModule


class HandcraftedPointsModule(LocomotionModule):
    """A locomotion module using target points with alternating leg groups."""

    class LegCyclePhase(Enum):
        """Leg cycle phase enum."""

        Lifting = auto()
        Reaching = auto()
        Planting = auto()
        Passing = auto()

    def __init__(self,
                 locomotion_node,
                 spiderbot_description,
                 phase_shift=1):
        """Initialize the locomotion module."""
        super().__init__(locomotion_node, spiderbot_description)

        # Add up the segment lengths
        self.leg_lengths = {}
        for leg_name in self.leg_names:
            leg_length = 0
            for segment_length in self.segment_lengths_per_leg[leg_name]:
                leg_length += segment_length
            self.leg_lengths[leg_name] = leg_length

        self.phase_length_seconds = 1.5
        self.phase_time_remaining = self.phase_length_seconds

        self.phase_shift = phase_shift % 4
        self.current_phase_group_1 = self.LegCyclePhase.Lifting
        match self.phase_shift:
            case 0:
                self.current_phase_group_2 = self.LegCyclePhase.Lifting
            case 1:
                self.current_phase_group_2 = self.LegCyclePhase.Reaching
            case 2:
                self.current_phase_group_2 = self.LegCyclePhase.Planting
            case 3:
                self.current_phase_group_2 = self.LegCyclePhase.Passing
            case _:  # Should be prevented by the modulus
                raise ValueError('phase_shift should be [0, 3]')

        front_l_lifting_targets = [-0.20, 0.70, -0.40]
        front_r_lifting_targets = [0.20, 0.70, -0.40]
        front_l_reaching_targets = [0.50, 0.70, -0.40]
        front_r_reaching_targets = [-0.50, 0.70, -0.40]
        front_l_planting_targets = [0.50, 0.40, -0.70]
        front_r_planting_targets = [-0.50, 0.40, -0.70]
        front_l_passing_targets = [-0.20, 0.60, -0.70]
        front_r_passing_targets = [0.20, 0.60, -0.70]

        back_l_lifting_targets = [-0.60, 0.80, -0.30]
        back_r_lifting_targets = [0.60, 0.80, -0.30]
        back_l_reaching_targets = [0.30, 0.90, -0.30]
        back_r_reaching_targets = [-0.30, 0.90, -0.30]
        back_l_planting_targets = [0.30, 0.60, -0.70]
        back_r_planting_targets = [-0.30, 0.60, -0.70]
        back_l_passing_targets = [-0.50, 0.70, -0.60]
        back_r_passing_targets = [0.50, 0.70, -0.60]

        lifting_targets = {
            'l_i': front_l_lifting_targets,
            'l_ii': front_l_lifting_targets,
            'l_iii': back_l_lifting_targets,
            'l_iv': back_l_lifting_targets,
            'r_i': front_r_lifting_targets,
            'r_ii': front_r_lifting_targets,
            'r_iii': back_r_lifting_targets,
            'r_iv': back_r_lifting_targets,
        }
        reaching_targets = {
            'l_i': front_l_reaching_targets,
            'l_ii': front_l_reaching_targets,
            'l_iii': back_l_reaching_targets,
            'l_iv': back_l_reaching_targets,
            'r_i': front_r_reaching_targets,
            'r_ii': front_r_reaching_targets,
            'r_iii': back_r_reaching_targets,
            'r_iv': back_r_reaching_targets,
        }
        planting_targets = {
            'l_i': front_l_planting_targets,
            'l_ii': front_l_planting_targets,
            'l_iii': back_l_planting_targets,
            'l_iv': back_l_planting_targets,
            'r_i': front_r_planting_targets,
            'r_ii': front_r_planting_targets,
            'r_iii': back_r_planting_targets,
            'r_iv': back_r_planting_targets,
        }
        passing_targets = {
            'l_i': front_l_passing_targets,
            'l_ii': front_l_passing_targets,
            'l_iii': back_l_passing_targets,
            'l_iv': back_l_passing_targets,
            'r_i': front_r_passing_targets,
            'r_ii': front_r_passing_targets,
            'r_iii': back_r_passing_targets,
            'r_iv': back_r_passing_targets,
        }
        self.targets = {
            'lifting': lifting_targets,
            'reaching': reaching_targets,
            'planting': planting_targets,
            'passing': passing_targets,
        }
        self.target_strings = {
            self.LegCyclePhase.Lifting: 'lifting',
            self.LegCyclePhase.Reaching: 'reaching',
            self.LegCyclePhase.Planting: 'planting',
            self.LegCyclePhase.Passing: 'passing',
        }

        self.previous_leg_targets = {}
        self.next_leg_targets = {}
        self.current_target_points = {}
        self.current_target_angles = {}

        self.group_1 = ['l_i', 'l_iii', 'r_ii', 'r_iv']
        self.group_2 = ['l_ii', 'l_iv', 'r_i', 'r_iii']

        # Group 1 (li, lii, rii, riv)
        targets_group_1 = self.targets[
            self.target_strings[self.current_phase_group_1]]
        for leg_name in self.group_1:
            self.previous_leg_targets[leg_name] = targets_group_1[leg_name]
            self.next_leg_targets[leg_name] = targets_group_1[leg_name]
            self.current_target_points[leg_name] = targets_group_1[leg_name]

        # Group 2 (lii, liv, ri, riii)
        targets_group_2 = self.targets[
            self.target_strings[self.current_phase_group_2]]
        for leg_name in self.group_2:
            self.previous_leg_targets[leg_name] = targets_group_2[leg_name]
            self.next_leg_targets[leg_name] = targets_group_2[leg_name]
            self.current_target_points[leg_name] = targets_group_2[leg_name]

    def _print_primary_phase(self):
        """Print the current primary phase."""
        match self.current_phase_group_1:
            case self.LegCyclePhase.Lifting:
                print('Lifting')
            case self.LegCyclePhase.Reaching:
                print('Passing')
            case self.LegCyclePhase.Planting:
                print('Planting')
            case self.LegCyclePhase.Passing:
                print('Passing')

    def _print_secondary_phase(self):
        """Print the current secondary phase."""
        match self.current_phase_group_2:
            case self.LegCyclePhase.Lifting:
                print('Lifting')
            case self.LegCyclePhase.Reaching:
                print('Passing')
            case self.LegCyclePhase.Planting:
                print('Planting')
            case self.LegCyclePhase.Passing:
                print('Passing')

    def _get_next_phase(self, state):
        """Get the next phase based on the current one."""
        match state:
            case self.LegCyclePhase.Lifting:
                return self.LegCyclePhase.Reaching
            case self.LegCyclePhase.Reaching:
                return self.LegCyclePhase.Planting
            case self.LegCyclePhase.Planting:
                return self.LegCyclePhase.Passing
            case self.LegCyclePhase.Passing:
                return self.LegCyclePhase.Lifting

    def _interpolate_leg_to_target(self,
                                   leg_name,
                                   leg_length,
                                   previous_leg_targets,
                                   next_leg_targets,
                                   percentage):
        """Interpolates the targets based on how far into the phase it is."""
        target_pos = np.asarray(previous_leg_targets) + (
                (
                    np.asarray(next_leg_targets) -
                    np.asarray(previous_leg_targets)
                )
                * percentage
            )
        # Scale to the leg
        scaled_target_pos = np.multiply(target_pos, leg_length)
        self.current_target_points[leg_name] = scaled_target_pos

    def _walk_forward(self, delta_time):
        """Walk the Spiderbot forward."""
        self.phase_time_remaining -= delta_time
        if self.phase_time_remaining < 0.0:
            self.phase_time_remaining = self.phase_length_seconds

            self.current_phase_group_1 = self._get_next_phase(
                self.current_phase_group_1)
            self.current_phase_group_2 = self._get_next_phase(
                self.current_phase_group_2)

            for leg_name in self.leg_names:
                self.previous_leg_targets[leg_name] = (
                    self.next_leg_targets[leg_name]
                )

            targets_group_1 = self.targets[
                self.target_strings[self.current_phase_group_1]]
            for leg_name in self.group_1:
                self.next_leg_targets[leg_name] = (
                    targets_group_1[leg_name]
                )
            targets_group_2 = self.targets[
                self.target_strings[self.current_phase_group_2]]
            for leg_name in self.group_2:
                self.next_leg_targets[leg_name] = (
                    targets_group_2[leg_name]
                )

        percentage = (
            1.0 - min(1.0,
                      max(0.0,
                          self.phase_time_remaining /
                          self.phase_length_seconds))
        )
        for leg_name in self.leg_names:
            self._interpolate_leg_to_target(
                leg_name,
                self.leg_lengths[leg_name],
                self.previous_leg_targets[leg_name],
                self.next_leg_targets[leg_name],
                percentage)

        self._convert_points_to_angles()

        self.publish_targets_and_angles()

    def _convert_points_to_angles(self):
        """Convert the target points to target angles."""
        for leg_name in self.leg_names:
            current_target_point = (
                self.current_target_points[leg_name]
            )
            self.legs[leg_name].move_claw_to_cartesian(
                current_target_point
            )
        mujoco.mj_step(self.model, self.data)

    def update(self, spiderbot_pose_msg):
        """Walk the spiderbot forward."""
        delta_time = self.get_delta_time_from_msg(spiderbot_pose_msg)
        if delta_time > 0.0:
            self._walk_forward(delta_time)

    def publish_targets_and_angles(self):
        """Publish target angles for the leg actuators."""
        timestamp = time.time()
        points_msg = utils.construct_leg_targets_msg(
            timestamp,
            self.leg_names,
            self.current_target_points
        )
        self.locomotion_node.publish_points(points_msg)
        angles_msg = utils.construct_target_pose_msg_from_legs(
                    timestamp,
                    self.leg_names,
                    self.legs
                )
        self.locomotion_node.publish_angles(angles_msg)
