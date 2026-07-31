from enum import Enum, auto

import numpy as np

from .locomotion import LocomotionModule

class MoveToPointLocomotionModule(LocomotionModule):

    class LegCyclePhase(Enum):
        Lifting = auto()
        Reaching = auto()
        Planting = auto()
        Passing = auto()
    
    def __init__(self, leg_set, phase_shift=1):
        self.phase_length = 1.5
        self.phase_shift = phase_shift % 4
        
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
    
        leg_l_i_start = [1.20, 0.80, -1.40]
        leg_r_i_start = [-1.20, 0.80, -1.40]
        leg_l_ii_start = [1.20, 0.80, -1.40]
        leg_r_ii_start = [-1.20, 0.80, -1.40]
        leg_l_iii_start = [1.20, 0.80, -1.40]
        leg_r_iii_start = [-1.20, 0.80, -1.40]
        leg_l_iv_start = [1.20, 0.80, -1.40]
        leg_r_iv_start = [-1.20, 0.80, -1.40]
        
        self.current_primary_state = self.LegCyclePhase.Lifting
        match self.phase_shift:
            case 0:
                self.current_secondary_state = self.LegCyclePhase.Lifting
            case 1:
                self.current_secondary_state = self.LegCyclePhase.Reaching
            case 2:
                self.current_secondary_state = self.LegCyclePhase.Planting
            case 3:
                self.current_secondary_state = self.LegCyclePhase.Passing
            case _:
                raise ValueError("phase_shift should be [0, 3]") # Should be prevented by the modulus
        self.timer = self.phase_length

        lifting_targets = {
            "l_i" : front_l_lifting_targets,
            "l_ii" : front_l_lifting_targets,
            "l_iii" : back_l_lifting_targets,
            "l_iv" : back_l_lifting_targets,
            "r_i" : front_r_lifting_targets,
            "r_ii" : front_r_lifting_targets,
            "r_iii" : back_r_lifting_targets,
            "r_iv" : back_r_lifting_targets,
        }
        reaching_targets = {
                    "l_i" : front_l_reaching_targets,
                    "l_ii" : front_l_reaching_targets,
                    "l_iii" : back_l_reaching_targets,
                    "l_iv" : back_l_reaching_targets,
                    "r_i" : front_r_reaching_targets,
                    "r_ii" : front_r_reaching_targets,
                    "r_iii" : back_r_reaching_targets,
                    "r_iv" : back_r_reaching_targets,
                }
        planting_targets = {
                    "l_i" : front_l_planting_targets,
                    "l_ii" : front_l_planting_targets,
                    "l_iii" : back_l_planting_targets,
                    "l_iv" : back_l_planting_targets,
                    "r_i" : front_r_planting_targets,
                    "r_ii" : front_r_planting_targets,
                    "r_iii" : back_r_planting_targets,
                    "r_iv" : back_r_planting_targets,
                }
        passing_targets = {
                    "l_i" : front_l_passing_targets,
                    "l_ii" : front_l_passing_targets,
                    "l_iii" : back_l_passing_targets,
                    "l_iv" : back_l_passing_targets,
                    "r_i" : front_r_passing_targets,
                    "r_ii" : front_r_passing_targets,
                    "r_iii" : back_r_passing_targets,
                    "r_iv" : back_r_passing_targets,
                }
        self.targets = {
            "lifting" : lifting_targets,
            "reaching" : reaching_targets,
            "planting" : planting_targets,
            "passing" : passing_targets,
        }
        self.target_strings = {
            self.LegCyclePhase.Lifting : "lifting",
            self.LegCyclePhase.Reaching : "reaching",
            self.LegCyclePhase.Planting : "planting",
            self.LegCyclePhase.Passing : "passing",
        }

        self.left_i_leg_targets = leg_l_i_start
        self.left_ii_leg_targets = leg_l_ii_start
        self.right_iii_leg_targets = leg_r_iii_start
        self.left_iv_leg_targets = leg_l_iv_start
        self.right_i_leg_targets = leg_r_i_start
        self.right_ii_leg_targets = leg_r_ii_start
        self.left_iii_leg_targets = leg_l_iii_start
        self.right_iv_leg_targets = leg_r_iv_start
        
        leg_set.left_i_leg.move_claw_to_cartesian(leg_l_i_start)
        leg_set.left_ii_leg.move_claw_to_cartesian(leg_l_ii_start)
        leg_set.left_iii_leg.move_claw_to_cartesian(leg_l_iii_start)
        leg_set.left_iv_leg.move_claw_to_cartesian(leg_l_iv_start)

        leg_set.right_i_leg.move_claw_to_cartesian(leg_r_i_start)
        leg_set.right_ii_leg.move_claw_to_cartesian(leg_r_ii_start)
        leg_set.right_iii_leg.move_claw_to_cartesian(leg_r_iii_start)
        leg_set.right_iv_leg.move_claw_to_cartesian(leg_r_iv_start)

        # Group 1 (li, lii, rii, riv)

        self.left_i_next_leg_targets = self.targets[self.target_strings[self.current_primary_state]]["l_i"]
        self.left_iii_next_leg_targets = self.targets[self.target_strings[self.current_primary_state]]["l_iii"]
        self.right_ii_next_leg_targets = self.targets[self.target_strings[self.current_primary_state]]["r_ii"]
        self.right_iv_next_leg_targets = self.targets[self.target_strings[self.current_primary_state]]["r_iv"]

        # Group 2 (lii, liv, ri, riii)

        self.left_ii_next_leg_targets = self.targets[self.target_strings[self.current_secondary_state]]["l_ii"]
        self.left_iv_next_leg_targets = self.targets[self.target_strings[self.current_secondary_state]]["l_iv"]
        self.right_i_next_leg_targets = self.targets[self.target_strings[self.current_secondary_state]]["r_ii"]
        self.right_iii_next_leg_targets = self.targets[self.target_strings[self.current_secondary_state]]["r_iii"]

    def print_primary_cycle(self):
        match self.current_primary_state:
            case self.LegCyclePhase.Lifting:
                print("Lifting")
            case self.LegCyclePhase.Reaching:
                print("Passing")
            case self.LegCyclePhase.Planting:
                print("Planting")
            case self.LegCyclePhase.Passing:
                print("Passing")

    def get_next_phase(self, state):
        match state:
            case self.LegCyclePhase.Lifting:
                return self.LegCyclePhase.Reaching
            case self.LegCyclePhase.Reaching:
                return self.LegCyclePhase.Planting
            case self.LegCyclePhase.Planting:
                return self.LegCyclePhase.Passing                        
            case self.LegCyclePhase.Passing:
                return self.LegCyclePhase.Lifting

    def interpolate_leg_to_target(self, leg, leg_targets, next_leg_targets, percentage):
        target_pos = np.asarray(leg_targets) + ((np.asarray(next_leg_targets) - np.asarray(leg_targets)) * percentage)
        scaled_target_pos = np.multiply(target_pos, leg.leg_length) # Scale to the leg
        leg.move_claw_to_cartesian(scaled_target_pos)

    def walk_forward(self, delta_time, leg_set):
        self.timer -= delta_time

        if self.timer < 0.0:
            self.timer = self.phase_length

            self.left_i_leg_targets = self.left_i_next_leg_targets
            self.left_ii_leg_targets = self.left_ii_next_leg_targets
            self.left_iii_leg_targets = self.left_iii_next_leg_targets
            self.left_iv_leg_targets = self.left_iv_next_leg_targets

            self.right_i_leg_targets = self.right_i_next_leg_targets
            self.right_ii_leg_targets = self.right_ii_next_leg_targets
            self.right_iii_leg_targets = self.right_iii_next_leg_targets
            self.right_iv_leg_targets = self.right_iv_next_leg_targets

            self.current_primary_state = self.get_next_phase(self.current_primary_state)
            self.current_secondary_state = self.get_next_phase(self.current_secondary_state)

            self.left_i_next_leg_targets = self.targets[self.target_strings[self.current_primary_state]]["l_i"]
            self.left_iii_next_leg_targets = self.targets[self.target_strings[self.current_primary_state]]["l_iii"]
            self.right_ii_next_leg_targets = self.targets[self.target_strings[self.current_primary_state]]["r_ii"]
            self.right_iv_next_leg_targets = self.targets[self.target_strings[self.current_primary_state]]["r_iv"]

            self.left_ii_next_leg_targets = self.targets[self.target_strings[self.current_secondary_state]]["l_ii"]
            self.left_iv_next_leg_targets = self.targets[self.target_strings[self.current_secondary_state]]["l_iv"]
            self.right_i_next_leg_targets = self.targets[self.target_strings[self.current_secondary_state]]["r_ii"]
            self.right_iii_next_leg_targets = self.targets[self.target_strings[self.current_secondary_state]]["r_iii"]

        percentage = 1.0 - min(1.0, max(0.0, self.timer / self.phase_length))

        self.interpolate_leg_to_target(leg_set.left_i_leg, self.left_i_leg_targets, self.left_i_next_leg_targets, percentage)
        self.interpolate_leg_to_target(leg_set.left_ii_leg, self.left_ii_leg_targets, self.left_ii_next_leg_targets, percentage)
        self.interpolate_leg_to_target(leg_set.left_iii_leg, self.left_iii_leg_targets, self.left_iii_next_leg_targets, percentage)
        self.interpolate_leg_to_target(leg_set.left_iv_leg, self.left_iv_leg_targets, self.left_iv_next_leg_targets, percentage)

        self.interpolate_leg_to_target(leg_set.right_i_leg, self.right_i_leg_targets, self.right_i_next_leg_targets, percentage)
        self.interpolate_leg_to_target(leg_set.right_ii_leg, self.right_ii_leg_targets, self.right_ii_next_leg_targets, percentage)
        self.interpolate_leg_to_target(leg_set.right_iii_leg, self.right_iii_leg_targets, self.right_iii_next_leg_targets, percentage)
        self.interpolate_leg_to_target(leg_set.right_iv_leg, self.right_iv_leg_targets, self.right_iv_next_leg_targets, percentage)