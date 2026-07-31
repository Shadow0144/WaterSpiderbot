import math
import numpy as np
from enum import Enum, auto

from .locomotion import LocomotionModule

phase_timer = 1.5

front_l_lifting_targets = [-0.40, 1.20, -0.80]
front_r_lifting_targets = [0.40, 1.20, -0.80]
front_l_reaching_targets = [1.00, 1.20, -0.80]
front_r_reaching_targets = [-1.00, 1.20, -0.80]
front_l_planting_targets = [1.00, 0.80, -1.40]
front_r_planting_targets = [-1.00, 0.80, -1.40]
front_l_passing_targets = [-0.40, 1.20, -1.20]
front_r_passing_targets = [0.40, 1.20, -1.20]

back_l_lifting_targets = [-1.20, 1.20, -0.80]
back_r_lifting_targets = [1.20, 1.20, -0.80]
back_l_reaching_targets = [0.40, 1.40, -0.60]
back_r_reaching_targets = [-0.40, 1.40, -0.60]
back_l_planting_targets = [0.40, 1.40, -1.20]
back_r_planting_targets = [-0.40, 1.40, -1.20]
back_l_passing_targets = [-1.00, 1.20, -0.80]
back_r_passing_targets = [1.00, 1.20, -0.80]

leg_l_i_start = [1.20, 0.80, -1.40]
leg_r_i_start = [-1.20, 0.80, -1.40]
leg_l_ii_start = [1.20, 0.80, -1.40]
leg_r_ii_start = [-1.20, 0.80, -1.40]
leg_l_iii_start = [1.20, 0.80, -1.40]
leg_r_iii_start = [-1.20, 0.80, -1.40]
leg_l_iv_start = [1.20, 0.80, -1.40]
leg_r_iv_start = [-1.20, 0.80, -1.40]

class MoveToPointLocomotionModule(LocomotionModule):

    class LegCycleState(Enum):
        Lifting_Planting = auto()
        Reaching_Passing = auto()
        Planting_Lifting = auto()
        Passing_Reaching = auto()
    
    def __init__(self, leg_set):
        self.current_state = self.LegCycleState.Lifting_Planting
        self.timer = phase_timer
        self.time_to_complete = phase_timer

        # Group 1 (li, lii, rii, riv)

        self.left_i_leg_targets = leg_l_i_start
        self.left_i_next_leg_targets = front_l_lifting_targets
        self.left_iii_leg_targets = leg_l_iii_start
        self.left_iii_next_leg_targets = back_l_lifting_targets
        
        self.right_ii_leg_targets = leg_r_ii_start
        self.right_ii_next_leg_targets = front_r_lifting_targets
        self.right_iv_leg_targets = leg_r_iv_start
        self.right_iv_next_leg_targets = back_r_lifting_targets

        # Group 2 (lii, liv, ri, riii)

        self.left_ii_leg_targets = leg_l_ii_start
        self.left_ii_next_leg_targets = front_l_passing_targets
        self.left_iv_leg_targets = leg_l_iv_start
        self.left_iv_next_leg_targets = back_l_passing_targets

        self.right_i_leg_targets = leg_r_i_start
        self.right_i_next_leg_targets = front_l_passing_targets
        self.right_iii_leg_targets = leg_r_iii_start
        self.right_iii_next_leg_targets = back_r_passing_targets
        
        leg_set.left_i_leg.move_claw_to_cartesian(leg_l_i_start)
        leg_set.left_ii_leg.move_claw_to_cartesian(leg_l_ii_start)
        leg_set.left_iii_leg.move_claw_to_cartesian(leg_l_iii_start)
        leg_set.left_iv_leg.move_claw_to_cartesian(leg_l_iv_start)
        leg_set.right_i_leg.move_claw_to_cartesian(leg_r_i_start)
        leg_set.right_ii_leg.move_claw_to_cartesian(leg_r_ii_start)
        leg_set.right_iii_leg.move_claw_to_cartesian(leg_r_iii_start)
        leg_set.right_iv_leg.move_claw_to_cartesian(leg_r_iv_start)

    def leg_cycle(self, leg, leg_targets, next_leg_targets):
        percentage = self.timer / self.time_to_complete
        target_pos = ((np.asarray(leg_targets) * percentage) + (np.asarray(next_leg_targets) * (1.0 - percentage)))
        target_pos = np.multiply(target_pos, leg.leg_length) # Scale to the leg
        leg.move_claw_to_cartesian(target_pos)

    def print_cycle(self):
        match self.current_state:
            case self.LegCycleState.Lifting_Planting:
                print("Lifting")

            case self.LegCycleState.Reaching_Passing:
                print("Passing")

            case self.LegCycleState.Planting_Lifting:
                print("Planting")
                
            case self.LegCycleState.Passing_Reaching:
                print("Passing")

    def walk_forward(self, delta_time, leg_set):
        self.timer -= delta_time

        #self.print_cycle()

        if self.timer < 0:
            self.timer = phase_timer

            self.left_i_leg_targets = self.left_i_next_leg_targets
            self.left_ii_leg_targets = self.left_ii_next_leg_targets
            self.left_iii_leg_targets = self.left_iii_next_leg_targets
            self.left_iv_leg_targets = self.left_iv_next_leg_targets

            self.right_i_leg_targets = self.right_i_next_leg_targets
            self.right_ii_leg_targets = self.right_ii_next_leg_targets
            self.right_iii_leg_targets = self.right_iii_next_leg_targets
            self.right_iv_leg_targets = self.right_iv_next_leg_targets

            match self.current_state:
                    case self.LegCycleState.Lifting_Planting:
                        self.current_state = self.LegCycleState.Reaching_Passing

                        self.left_i_next_leg_targets = front_l_reaching_targets  
                        self.left_iii_next_leg_targets = back_l_reaching_targets
                        self.right_ii_next_leg_targets = front_r_reaching_targets 
                        self.right_iv_next_leg_targets = back_r_reaching_targets 

                        self.left_ii_next_leg_targets = front_l_passing_targets 
                        self.left_iv_next_leg_targets = back_l_passing_targets 
                        self.right_i_next_leg_targets = front_r_passing_targets  
                        self.right_iii_next_leg_targets = back_r_passing_targets

                    case self.LegCycleState.Reaching_Passing:
                        self.current_state = self.LegCycleState.Planting_Lifting

                        self.left_i_next_leg_targets = front_l_planting_targets  
                        self.left_iii_next_leg_targets = back_l_planting_targets
                        self.right_ii_next_leg_targets = front_r_planting_targets 
                        self.right_iv_next_leg_targets = back_r_planting_targets 
                        
                        self.left_ii_next_leg_targets = front_l_lifting_targets 
                        self.left_iv_next_leg_targets = back_l_lifting_targets 
                        self.right_i_next_leg_targets = front_r_lifting_targets  
                        self.right_iii_next_leg_targets = back_r_lifting_targets

                    case self.LegCycleState.Planting_Lifting:
                        self.current_state = self.LegCycleState.Passing_Reaching

                        self.left_i_next_leg_targets = front_l_passing_targets  
                        self.left_iii_next_leg_targets = back_l_passing_targets
                        self.right_ii_next_leg_targets = front_r_passing_targets 
                        self.right_iv_next_leg_targets = back_r_passing_targets 

                        self.left_ii_next_leg_targets = front_l_reaching_targets 
                        self.left_iv_next_leg_targets = back_l_reaching_targets 
                        self.right_i_next_leg_targets = front_r_reaching_targets  
                        self.right_iii_next_leg_targets = back_r_reaching_targets
                        
                    case self.LegCycleState.Passing_Reaching:
                        self.current_state = self.LegCycleState.Lifting_Planting

                        self.left_i_next_leg_targets = front_l_lifting_targets  
                        self.left_iii_next_leg_targets = back_l_lifting_targets
                        self.right_ii_next_leg_targets = front_r_lifting_targets 
                        self.right_iv_next_leg_targets = back_r_lifting_targets 

                        self.left_ii_next_leg_targets = front_l_planting_targets 
                        self.left_iv_next_leg_targets = back_l_planting_targets 
                        self.right_i_next_leg_targets = front_r_planting_targets  
                        self.right_iii_next_leg_targets = back_r_planting_targets

        self.leg_cycle(leg_set.left_i_leg, self.left_i_leg_targets, self.left_i_next_leg_targets)
        self.leg_cycle(leg_set.left_ii_leg, self.left_ii_leg_targets, self.left_ii_next_leg_targets)
        self.leg_cycle(leg_set.left_iii_leg, self.left_iii_leg_targets, self.left_iii_next_leg_targets)
        self.leg_cycle(leg_set.left_iv_leg, self.left_iv_leg_targets, self.left_iv_next_leg_targets)

        self.leg_cycle(leg_set.right_i_leg, self.right_i_leg_targets, self.right_i_next_leg_targets)
        self.leg_cycle(leg_set.right_ii_leg, self.right_ii_leg_targets, self.right_ii_next_leg_targets)
        self.leg_cycle(leg_set.right_iii_leg, self.right_iii_leg_targets, self.right_iii_next_leg_targets)
        self.leg_cycle(leg_set.right_iv_leg, self.right_iv_leg_targets, self.right_iv_next_leg_targets)