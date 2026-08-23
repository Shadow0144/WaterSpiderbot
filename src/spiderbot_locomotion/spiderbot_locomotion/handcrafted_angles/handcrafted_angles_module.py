"""Locomotion module using handcrafted angle targets."""

import math
import time
from enum import Enum, auto

import numpy as np

import spiderbot_utilities as utils

from ..locomotion_module import LocomotionModule


class HandcraftedAnglesModule(LocomotionModule):
    """Locomotion module using handcrafted angle targets."""

    class LegCyclePhase(Enum):
        """Leg cycle phase enum."""

        Lifting = auto()
        Reaching = auto()
        Planting = auto()
        Passing = auto()

    def __init__(self, locomotion_node,
                 spiderbot_description,
                 phase_shift=2):
        """Initialize the locomotion module."""
        super().__init__(locomotion_node, spiderbot_description)
        self.phase_length_seconds = 4.5
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

        front_lifting_targets = [math.radians(20),
                                 math.radians(30),
                                 math.radians(15)]
        front_reaching_targets = [math.radians(-20),
                                  math.radians(30),
                                  math.radians(15)]
        front_planting_targets = [math.radians(-20),
                                  math.radians(0),
                                  math.radians(0)]
        front_passing_targets = [math.radians(20),
                                 math.radians(0),
                                 math.radians(0)]

        back_lifting_targets = [math.radians(15),
                                math.radians(30),
                                math.radians(15)]
        back_reaching_targets = [math.radians(-30),
                                 math.radians(30),
                                 math.radians(15)]
        back_planting_targets = [math.radians(-30),
                                 math.radians(0),
                                 math.radians(0)]
        back_passing_targets = [math.radians(15),
                                math.radians(0),
                                math.radians(0)]

        leg_i_offsets = [math.radians(0),
                         math.radians(60),
                         math.radians(0)]
        leg_ii_offsets = [math.radians(0),
                          math.radians(15),
                          math.radians(0)]
        leg_iii_offsets = [math.radians(0),
                           math.radians(10),
                           math.radians(-10)]
        leg_iv_offsets = [math.radians(0),
                          math.radians(30),
                          math.radians(-30)]

        lifting_targets = {
            'l_i': front_lifting_targets + leg_i_offsets,
            'l_ii': front_lifting_targets + leg_ii_offsets,
            'l_iii': back_lifting_targets + leg_iii_offsets,
            'l_iv': back_lifting_targets + leg_iv_offsets,
            'r_i': front_lifting_targets + leg_i_offsets,
            'r_ii': front_lifting_targets + leg_ii_offsets,
            'r_iii': back_lifting_targets + leg_iii_offsets,
            'r_iv': back_lifting_targets + leg_iv_offsets,
        }
        reaching_targets = {
            'l_i': front_reaching_targets + leg_i_offsets,
            'l_ii': front_reaching_targets + leg_ii_offsets,
            'l_iii': back_reaching_targets + leg_iii_offsets,
            'l_iv': back_reaching_targets + leg_iv_offsets,
            'r_i': front_reaching_targets + leg_i_offsets,
            'r_ii': front_reaching_targets + leg_ii_offsets,
            'r_iii': back_reaching_targets + leg_iii_offsets,
            'r_iv': back_reaching_targets + leg_iv_offsets,
        }
        planting_targets = {
            'l_i': front_planting_targets + leg_i_offsets,
            'l_ii': front_planting_targets + leg_ii_offsets,
            'l_iii': back_planting_targets + leg_iii_offsets,
            'l_iv': back_planting_targets + leg_iv_offsets,
            'r_i': front_planting_targets + leg_i_offsets,
            'r_ii': front_planting_targets + leg_ii_offsets,
            'r_iii': back_planting_targets + leg_iii_offsets,
            'r_iv': back_planting_targets + leg_iv_offsets,
        }
        passing_targets = {
            'l_i': front_passing_targets + leg_i_offsets,
            'l_ii': front_passing_targets + leg_ii_offsets,
            'l_iii': back_passing_targets + leg_iii_offsets,
            'l_iv': back_passing_targets + leg_iv_offsets,
            'r_i': front_passing_targets + leg_i_offsets,
            'r_ii': front_passing_targets + leg_ii_offsets,
            'r_iii': back_passing_targets + leg_iii_offsets,
            'r_iv': back_passing_targets + leg_iv_offsets,
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
        self.current_targets = {}

        self.group_1 = ['l_i', 'l_iii', 'r_ii', 'r_iv']
        self.group_2 = ['l_ii', 'l_iv', 'r_i', 'r_iii']

        # Group 1 (li, lii, rii, riv)
        targets_group_1 = self.targets[
            self.target_strings[self.current_phase_group_1]]
        for leg_name in self.group_1:
            self.previous_leg_targets[leg_name] = targets_group_1[leg_name]
            self.next_leg_targets[leg_name] = targets_group_1[leg_name]
            self.current_targets[leg_name] = targets_group_1[leg_name]

        # Group 2 (lii, liv, ri, riii)
        targets_group_2 = self.targets[
            self.target_strings[self.current_phase_group_2]]
        for leg_name in self.group_2:
            self.previous_leg_targets[leg_name] = targets_group_2[leg_name]
            self.next_leg_targets[leg_name] = targets_group_2[leg_name]
            self.current_targets[leg_name] = targets_group_2[leg_name]

    def print_primary_cycle(self):
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

    def get_next_phase(self, state):
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

    def interpolate_leg_to_target(self,
                                  leg_name,
                                  previous_leg_targets,
                                  next_leg_targets,
                                  percentage):
        """Interpolate the target based on how far into the phase it is."""
        target_angles = np.asarray(previous_leg_targets) + (
            (np.asarray(next_leg_targets) -
             np.asarray(previous_leg_targets))
            * percentage)
        self.current_targets[leg_name] = [
            target_angles[0],
            target_angles[1],
            target_angles[2]
        ]

    def walk_forward(self, delta_time):
        """Walk the Spiderbot forward."""
        self.phase_time_remaining -= delta_time
        if self.phase_time_remaining < 0.0:
            self.phase_time_remaining = self.phase_length_seconds

            self.current_phase_group_1 = self.get_next_phase(
                self.current_phase_group_1)
            self.current_phase_group_2 = self.get_next_phase(
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
            self.interpolate_leg_to_target(
                leg_name,
                self.previous_leg_targets[leg_name],
                self.next_leg_targets[leg_name],
                percentage)

        self.publish_angles()

    def update(self, spiderbot_pose_msg):
        """Walk the spiderbot forward."""
        delta_time = self.get_delta_time_from_msg(spiderbot_pose_msg)
        if delta_time > 0.0:
            self.walk_forward(delta_time)

    def publish_angles(self):
        """Publish target angles for the leg actuators."""
        msg = utils.construct_target_pose_msg(
                    time.time(),
                    self.leg_names,
                    self.current_targets
                )
        self.locomotion_node.publish_angles(msg)
