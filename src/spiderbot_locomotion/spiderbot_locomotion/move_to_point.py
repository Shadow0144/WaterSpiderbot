"""A locomotion module using target points with alternating leg groups."""

from enum import Enum, auto

import numpy as np

from .locomotion import LocomotionModule


class MoveToPointLocomotionModule(LocomotionModule):
    """A locomotion module using target points with alternating leg groups."""

    class LegCyclePhase(Enum):
        """Leg cycle phase enum."""

        Lifting = auto()
        Reaching = auto()
        Planting = auto()
        Passing = auto()

    def __init__(self, phase_shift=1):
        """Initialize the locomotion module."""
        self.phase_length = 1.5
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
        self.timer = self.phase_length

        front_l_lifting_targets = [-0.40, 1.40, -0.80]
        front_r_lifting_targets = [0.40, 1.40, -0.80]
        front_l_reaching_targets = [1.00, 1.40, -0.80]
        front_r_reaching_targets = [-1.00, 1.40, -0.80]
        front_l_planting_targets = [1.00, 0.80, -1.40]
        front_r_planting_targets = [-1.00, 0.80, -1.40]
        front_l_passing_targets = [-0.40, 1.20, -1.40]
        front_r_passing_targets = [0.40, 1.20, -1.40]

        back_l_lifting_targets = [-1.20, 1.60, -0.60]
        back_r_lifting_targets = [1.20, 1.60, -0.60]
        back_l_reaching_targets = [0.60, 1.80, -0.60]
        back_r_reaching_targets = [-0.60, 1.80, -0.60]
        back_l_planting_targets = [0.60, 1.20, -1.40]
        back_r_planting_targets = [-0.60, 1.20, -1.40]
        back_l_passing_targets = [-1.00, 1.40, -1.20]
        back_r_passing_targets = [1.00, 1.40, -1.20]

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

        # Group 1 (li, lii, rii, riv)

        targets_group_1 = self.targets[
            self.target_strings[self.current_phase_group_1]]

        self.left_i_leg_targets = targets_group_1['l_i']
        self.left_i_next_leg_targets = targets_group_1['l_i']

        self.left_iii_leg_targets = targets_group_1['l_iii']
        self.left_iii_next_leg_targets = targets_group_1['l_iii']

        self.right_ii_leg_targets = targets_group_1['r_ii']
        self.right_ii_next_leg_targets = targets_group_1['r_ii']

        self.right_iv_leg_targets = targets_group_1['r_iv']
        self.right_iv_next_leg_targets = targets_group_1['r_iv']

        # Group 2 (lii, liv, ri, riii)

        targets_group_2 = self.targets[
            self.target_strings[self.current_phase_group_2]]

        self.left_ii_leg_targets = targets_group_2['l_ii']
        self.left_ii_next_leg_targets = targets_group_2['l_ii']

        self.left_iv_leg_targets = targets_group_2['l_iv']
        self.left_iv_next_leg_targets = targets_group_2['l_iv']

        self.right_i_leg_targets = targets_group_2['r_ii']
        self.right_i_next_leg_targets = targets_group_2['r_ii']

        self.right_iii_leg_targets = targets_group_2['r_iii']
        self.right_iii_next_leg_targets = targets_group_2['r_iii']

        self.current_targets = {
            'l_i': self.left_i_leg_targets,
            'l_ii': self.left_ii_leg_targets,
            'l_iii': self.left_iii_leg_targets,
            'l_iv': self.left_iv_leg_targets,
            'r_i': self.right_i_leg_targets,
            'r_ii': self.right_ii_leg_targets,
            'r_iii': self.right_iii_leg_targets,
            'r_iv': self.right_iv_leg_targets,
        }

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
                                  leg,
                                  leg_length,
                                  leg_targets,
                                  next_leg_targets,
                                  percentage):
        """Interpolates the targets based on how far into the phase it is."""
        target_pos = np.asarray(leg_targets) + (
            (np.asarray(next_leg_targets) - np.asarray(leg_targets))
            * percentage)
        # Scale to the leg
        scaled_target_pos = np.multiply(target_pos, leg_length)
        self.current_targets[leg] = scaled_target_pos

    def walk_forward(self, delta_time):
        """Walk the Spiderbot forward."""
        self.timer -= delta_time

        if self.timer < 0.0:
            self.timer = self.phase_length

            self.current_phase_group_1 = self.get_next_phase(
                self.current_phase_group_1)
            self.current_phase_group_2 = self.get_next_phase(
                self.current_phase_group_2)

            self.left_i_leg_targets = self.left_i_next_leg_targets
            self.left_ii_leg_targets = self.left_ii_next_leg_targets
            self.left_iii_leg_targets = self.left_iii_next_leg_targets
            self.left_iv_leg_targets = self.left_iv_next_leg_targets

            self.right_i_leg_targets = self.right_i_next_leg_targets
            self.right_ii_leg_targets = self.right_ii_next_leg_targets
            self.right_iii_leg_targets = self.right_iii_next_leg_targets
            self.right_iv_leg_targets = self.right_iv_next_leg_targets

            # Group 1 (li, lii, rii, riv)

            targets_group_1 = self.targets[
                self.target_strings[self.current_phase_group_1]]
            self.left_i_next_leg_targets = targets_group_1['l_i']
            self.left_iii_next_leg_targets = targets_group_1['l_iii']
            self.right_ii_next_leg_targets = targets_group_1['r_ii']
            self.right_iv_next_leg_targets = targets_group_1['r_iv']

            # Group 2 (lii, liv, ri, riii)

            targets_group_2 = self.targets[
                self.target_strings[self.current_phase_group_2]]
            self.left_ii_next_leg_targets = targets_group_2['l_ii']
            self.left_iv_next_leg_targets = targets_group_2['l_iv']
            self.right_i_next_leg_targets = targets_group_2['r_ii']
            self.right_iii_next_leg_targets = targets_group_2['r_iii']

        percentage = 1.0 - min(1.0, max(0.0, self.timer / self.phase_length))

        self.interpolate_leg_to_target('l_i',
                                       1.00 * 0.25,  # TODO
                                       self.left_i_leg_targets,
                                       self.left_i_next_leg_targets,
                                       percentage)
        self.interpolate_leg_to_target('l_ii',
                                       0.90 * 0.25,  # TODO
                                       self.left_ii_leg_targets,
                                       self.left_ii_next_leg_targets,
                                       percentage)
        self.interpolate_leg_to_target('l_iii',
                                       0.75 * 0.25,  # TODO
                                       self.left_iii_leg_targets,
                                       self.left_iii_next_leg_targets,
                                       percentage)
        self.interpolate_leg_to_target('l_iv',
                                       1.10 * 0.25,  # TODO
                                       self.left_iv_leg_targets,
                                       self.left_iv_next_leg_targets,
                                       percentage)

        self.interpolate_leg_to_target('r_i',
                                       1.00 * 0.25,  # TODO
                                       self.right_i_leg_targets,
                                       self.right_i_next_leg_targets,
                                       percentage)
        self.interpolate_leg_to_target('r_ii',
                                       0.90 * 0.25,  # TODO
                                       self.right_ii_leg_targets,
                                       self.right_ii_next_leg_targets,
                                       percentage)
        self.interpolate_leg_to_target('r_iii',
                                       0.75 * 0.25,  # TODO
                                       self.right_iii_leg_targets,
                                       self.right_iii_next_leg_targets,
                                       percentage)
        self.interpolate_leg_to_target('r_iv',
                                       1.10 * 0.25,  # TODO
                                       self.right_iv_leg_targets,
                                       self.right_iv_next_leg_targets,
                                       percentage)
