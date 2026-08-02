"""Locomotion module using handcrafted angle targets."""

import math
from enum import Enum, auto

from .locomotion import LocomotionModule


class HandcraftedLocomotionModule(LocomotionModule):
    """Locomotion module using handcrafted angle targets."""

    class LegCyclePhase(Enum):
        """Leg cycle phase enum."""

        Lifting_Planting = auto()
        Reaching_Passing = auto()
        Planting_Lifting = auto()
        Passing_Reaching = auto()

    def __init__(self, leg_set):
        """Initialize the locomotion module."""
        self.current_phase = self.LegCyclePhase.Lifting_Planting

        self.phase_length_seconds = 4.5
        self.timer = self.phase_length_seconds

        self.front_lifting_targets = [math.radians(20),
                                      math.radians(30),
                                      math.radians(15)]
        self.front_reaching_targets = [math.radians(-20),
                                       math.radians(30),
                                       math.radians(15)]
        self.front_planting_targets = [math.radians(-20),
                                       math.radians(0),
                                       math.radians(0)]
        self.front_passing_targets = [math.radians(20),
                                      math.radians(0),
                                      math.radians(0)]

        self.back_lifting_targets = [math.radians(15),
                                     math.radians(30),
                                     math.radians(15)]
        self.back_reaching_targets = [math.radians(-30),
                                      math.radians(30),
                                      math.radians(15)]
        self.back_planting_targets = [math.radians(-30),
                                      math.radians(0),
                                      math.radians(0)]
        self.back_passing_targets = [math.radians(15),
                                     math.radians(0),
                                     math.radians(0)]

        self.leg_i_offsets = [math.radians(0),
                              math.radians(60),
                              math.radians(0)]
        self.leg_ii_offsets = [math.radians(0),
                               math.radians(15),
                               math.radians(0)]
        self.leg_iii_offsets = [math.radians(0),
                                math.radians(10),
                                math.radians(-10)]
        self.leg_iv_offsets = [math.radians(0),
                               math.radians(30),
                               math.radians(-30)]

        # Group 1 (li, lii, rii, riv)

        self.left_i_leg_targets = self.leg_i_offsets
        self.left_i_next_leg_targets = (
            self.front_lifting_targets + self.leg_i_offsets
        )
        self.left_iii_leg_targets = self.leg_iii_offsets
        self.left_iii_next_leg_targets = (
            self.back_lifting_targets + self.leg_iii_offsets
        )

        self.right_ii_leg_targets = self.leg_ii_offsets
        self.right_ii_next_leg_targets = (
            self.front_lifting_targets + self.leg_ii_offsets
        )
        self.right_iv_leg_targets = self.leg_iv_offsets
        self.right_iv_next_leg_targets = (
            self.back_lifting_targets + self.leg_iv_offsets
        )

        # Group 2 (lii, liv, ri, riii)

        self.left_ii_leg_targets = self.leg_ii_offsets
        self.left_ii_next_leg_targets = (
            self.front_passing_targets + self.leg_ii_offsets
        )
        self.left_iv_leg_targets = self.leg_iv_offsets
        self.left_iv_next_leg_targets = (
            self.back_passing_targets + self.leg_iv_offsets
        )

        self.right_i_leg_targets = self.leg_i_offsets
        self.right_i_next_leg_targets = (
            self.front_passing_targets + self.leg_i_offsets
        )
        self.right_iii_leg_targets = self.leg_iii_offsets
        self.right_iii_next_leg_targets = (
            self.back_passing_targets + self.leg_iii_offsets
        )

        self.leg_cycle(leg_set.left_i_leg,
                       self.leg_i_offsets,
                       self.leg_i_offsets)
        self.leg_cycle(leg_set.left_ii_leg,
                       self.leg_ii_offsets,
                       self.leg_ii_offsets)
        self.leg_cycle(leg_set.left_iii_leg,
                       self.leg_iii_offsets,
                       self.leg_iii_offsets)
        self.leg_cycle(leg_set.left_iv_leg,
                       self.leg_iv_offsets,
                       self.leg_iv_offsets)

        self.leg_cycle(leg_set.right_i_leg,
                       self.leg_i_offsets,
                       self.leg_i_offsets)
        self.leg_cycle(leg_set.right_ii_leg,
                       self.leg_ii_offsets,
                       self.leg_ii_offsets)
        self.leg_cycle(leg_set.right_iii_leg,
                       self.leg_iii_offsets,
                       self.leg_iii_offsets)
        self.leg_cycle(leg_set.right_iv_leg,
                       self.leg_iv_offsets,
                       self.leg_iv_offsets)

    def interpolate_leg_to_target(self, leg, leg_targets, next_leg_targets):
        """Interpolates the targets based on how far into the phase it is."""
        percentage = self.timer / self.time_to_complete
        coxa_target_angle = ((leg_targets[0] * percentage) +
                             (next_leg_targets[0] * (1.0 - percentage)))
        femur_target_angle = ((leg_targets[1] * percentage) +
                              (next_leg_targets[1] * (1.0 - percentage)))
        tibia_target_angle = ((leg_targets[2] * percentage) +
                              (next_leg_targets[2] * (1.0 - percentage)))
        leg.set_leg_targets(coxa_target_angle,
                            femur_target_angle,
                            tibia_target_angle)

    def walk_forward(self, delta_time, leg_set):
        """Walk the Spiderbot forward."""
        self.timer -= delta_time

        if self.timer < 0:
            self.timer = self.phase_length_seconds

            self.left_i_leg_targets = self.left_i_next_leg_targets
            self.left_ii_leg_targets = self.left_ii_next_leg_targets
            self.left_iii_leg_targets = self.left_iii_next_leg_targets
            self.left_iv_leg_targets = self.left_iv_next_leg_targets

            self.right_i_leg_targets = self.right_i_next_leg_targets
            self.right_ii_leg_targets = self.right_ii_next_leg_targets
            self.right_iii_leg_targets = self.right_iii_next_leg_targets
            self.right_iv_leg_targets = self.right_iv_next_leg_targets

            match self.current_phase:
                case self.LegCyclePhase.Lifting_Planting:
                    self.current_phase = self.LegCyclePhase.Reaching_Passing

                    self.left_i_next_leg_targets = (
                        self.front_reaching_targets + self.leg_i_offsets
                    )
                    self.left_iii_next_leg_targets = (
                        self.back_reaching_targets + self.leg_iii_offsets
                    )
                    self.right_ii_next_leg_targets = (
                        self.front_reaching_targets + self.leg_ii_offsets
                    )
                    self.right_iv_next_leg_targets = (
                        self.back_reaching_targets + self.leg_iv_offsets
                    )

                    self.left_ii_next_leg_targets = (
                        self.front_passing_targets + self.leg_ii_offsets
                    )
                    self.left_iv_next_leg_targets = (
                        self.back_passing_targets + self.leg_iv_offsets
                    )
                    self.right_i_next_leg_targets = (
                        self.front_passing_targets + self.leg_i_offsets
                    )
                    self.right_iii_next_leg_targets = (
                        self.back_passing_targets + self.leg_iii_offsets
                    )

                case self.LegCyclePhase.Reaching_Passing:
                    self.current_phase = self.LegCyclePhase.Planting_Lifting

                    self.left_i_next_leg_targets = (
                        self.front_planting_targets + self.leg_i_offsets
                    )
                    self.left_iii_next_leg_targets = (
                        self.back_planting_targets + self.leg_iii_offsets
                    )
                    self.right_ii_next_leg_targets = (
                        self.front_planting_targets + self.leg_ii_offsets
                    )
                    self.right_iv_next_leg_targets = (
                        self.back_planting_targets + self.leg_iv_offsets
                    )

                    self.left_ii_next_leg_targets = (
                        self.front_lifting_targets + self.leg_ii_offsets
                    )
                    self.left_iv_next_leg_targets = (
                        self.back_lifting_targets + self.leg_iv_offsets
                    )
                    self.right_i_next_leg_targets = (
                        self.front_lifting_targets + self.leg_i_offsets
                    )
                    self.right_iii_next_leg_targets = (
                        self.back_lifting_targets + self.leg_iii_offsets
                    )

                case self.LegCyclePhase.Planting_Lifting:
                    self.current_phase = self.LegCyclePhase.Passing_Reaching

                    self.left_i_next_leg_targets = (
                        self.front_passing_targets + self.leg_i_offsets
                    )
                    self.left_iii_next_leg_targets = (
                        self.back_passing_targets + self.leg_iii_offsets
                    )
                    self.right_ii_next_leg_targets = (
                        self.front_passing_targets + self.leg_ii_offsets
                    )
                    self.right_iv_next_leg_targets = (
                        self.back_passing_targets + self.leg_iv_offsets
                    )

                    self.left_ii_next_leg_targets = (
                        self.front_reaching_targets + self.leg_ii_offsets
                    )
                    self.left_iv_next_leg_targets = (
                        self.back_reaching_targets + self.leg_iv_offsets
                    )
                    self.right_i_next_leg_targets = (
                        self.front_reaching_targets + self.leg_i_offsets
                    )
                    self.right_iii_next_leg_targets = (
                        self.back_reaching_targets + self.leg_iii_offsets
                    )

                case self.LegCyclePhase.Passing_Reaching:
                    self.current_phase = self.LegCyclePhase.Lifting_Planting

                    self.left_i_next_leg_targets = (
                        self.front_lifting_targets + self.leg_i_offsets
                    )
                    self.left_iii_next_leg_targets = (
                        self.back_lifting_targets + self.leg_iii_offsets
                    )
                    self.right_ii_next_leg_targets = (
                        self.front_lifting_targets + self.leg_ii_offsets
                    )
                    self.right_iv_next_leg_targets = (
                        self.back_lifting_targets + self.leg_iv_offsets
                    )

                    self.left_ii_next_leg_targets = (
                        self.front_planting_targets + self.leg_ii_offsets
                    )
                    self.left_iv_next_leg_targets = (
                        self.back_planting_targets + self.leg_iv_offsets
                    )
                    self.right_i_next_leg_targets = (
                        self.front_planting_targets + self.leg_i_offsets
                    )
                    self.right_iii_next_leg_targets = (
                        self.back_planting_targets + self.leg_iii_offsets
                    )

        self.interpolate_leg_to_target(leg_set.left_i_leg,
                                       self.left_i_leg_targets,
                                       self.left_i_next_leg_targets)
        self.interpolate_leg_to_target(leg_set.left_ii_leg,
                                       self.left_ii_leg_targets,
                                       self.left_ii_next_leg_targets)
        self.interpolate_leg_to_target(leg_set.left_iii_leg,
                                       self.left_iii_leg_targets,
                                       self.left_iii_next_leg_targets)
        self.interpolate_leg_to_target(leg_set.left_iv_leg,
                                       self.left_iv_leg_targets,
                                       self.left_iv_next_leg_targets)

        self.interpolate_leg_to_target(leg_set.right_i_leg,
                                       self.right_i_leg_targets,
                                       self.right_i_next_leg_targets)
        self.interpolate_leg_to_target(leg_set.right_ii_leg,
                                       self.right_ii_leg_targets,
                                       self.right_ii_next_leg_targets)
        self.interpolate_leg_to_target(leg_set.right_iii_leg,
                                       self.right_iii_leg_targets,
                                       self.right_iii_next_leg_targets)
        self.interpolate_leg_to_target(leg_set.right_iv_leg,
                                       self.right_iv_leg_targets,
                                       self.right_iv_next_leg_targets)
