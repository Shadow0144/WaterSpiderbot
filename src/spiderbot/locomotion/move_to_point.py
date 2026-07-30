import math
import numpy as np
from enum import Enum, auto

from .locomotion import LocomotionModule

phase_timer = 3

front_lifting_targets = [0.0, 0.35, 0.0]
front_reaching_targets = [0.15, 0.35, 0.0]
front_planting_targets = [0.15, -0.55, -0.25]
front_passing_targets = [0.0, -0.55, -0.25]

back_lifting_targets = [-0.5, 0.65, 0.0]
back_reaching_targets = [0.0, 0.65, 0.0]
back_planting_targets = [0.0, 0.0, -0.5]
back_passing_targets = [-0.5, 0.0, -0.5]

leg_i_start = [0.0, 0.65, 0.0]
leg_ii_start = [0.0, 0.65, 0.0]
leg_iii_start = [0.0, 0.65, 0.0]
leg_iv_start = [0.0, 0.65, 0.0]

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

        self.left_i_leg_targets = leg_i_start
        self.left_i_next_leg_targets = front_lifting_targets
        self.left_iii_leg_targets = leg_iii_start
        self.left_iii_next_leg_targets = back_lifting_targets
        
        self.right_ii_leg_targets = leg_ii_start
        self.right_ii_next_leg_targets = front_lifting_targets
        self.right_iv_leg_targets = leg_iv_start
        self.right_iv_next_leg_targets = back_lifting_targets

        # Group 2 (lii, liv, ri, riii)

        self.left_ii_leg_targets = leg_ii_start
        self.left_ii_next_leg_targets = front_passing_targets
        self.left_iv_leg_targets = leg_iv_start
        self.left_iv_next_leg_targets = back_passing_targets

        self.right_i_leg_targets = leg_i_start
        self.right_i_next_leg_targets = front_passing_targets
        self.right_iii_leg_targets = leg_iii_start
        self.right_iii_next_leg_targets = back_passing_targets
        
        leg_set.left_i_leg.move_claw_to_cartesian(leg_i_start)
        leg_set.left_ii_leg.move_claw_to_cartesian(leg_ii_start)
        leg_set.left_iii_leg.move_claw_to_cartesian(leg_iii_start)
        leg_set.left_iv_leg.move_claw_to_cartesian(leg_iv_start)
        leg_set.right_i_leg.move_claw_to_cartesian(leg_i_start)
        leg_set.right_ii_leg.move_claw_to_cartesian(leg_ii_start)
        leg_set.right_iii_leg.move_claw_to_cartesian(leg_iii_start)
        leg_set.right_iv_leg.move_claw_to_cartesian(leg_iv_start)

        self.leg_cycle(leg_set.left_i_leg, leg_i_start, leg_i_start)
        self.leg_cycle(leg_set.left_ii_leg, leg_ii_start, leg_ii_start)
        self.leg_cycle(leg_set.left_iii_leg, leg_iii_start, leg_iii_start)
        self.leg_cycle(leg_set.left_iv_leg, leg_iv_start, leg_iv_start)

        self.leg_cycle(leg_set.right_i_leg, leg_i_start, leg_i_start)
        self.leg_cycle(leg_set.right_ii_leg, leg_ii_start, leg_ii_start)
        self.leg_cycle(leg_set.right_iii_leg, leg_iii_start, leg_iii_start)
        self.leg_cycle(leg_set.right_iv_leg, leg_iv_start, leg_iv_start)

    def leg_cycle(self, leg, leg_targets, next_leg_targets):
        percentage = self.timer / self.time_to_complete
        target_pos = ((np.asarray(leg_targets) * percentage) + (np.asarray(next_leg_targets) * (1.0 - percentage)))
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

                        self.left_i_next_leg_targets = front_reaching_targets  
                        self.left_iii_next_leg_targets = back_reaching_targets
                        self.right_ii_next_leg_targets = front_reaching_targets 
                        self.right_iv_next_leg_targets = back_reaching_targets 

                        self.left_ii_next_leg_targets = front_passing_targets 
                        self.left_iv_next_leg_targets = back_passing_targets 
                        self.right_i_next_leg_targets = front_passing_targets  
                        self.right_iii_next_leg_targets = back_passing_targets

                    case self.LegCycleState.Reaching_Passing:
                        self.current_state = self.LegCycleState.Planting_Lifting

                        self.left_i_next_leg_targets = front_planting_targets  
                        self.left_iii_next_leg_targets = back_planting_targets
                        self.right_ii_next_leg_targets = front_planting_targets 
                        self.right_iv_next_leg_targets = back_planting_targets 
                        
                        self.left_ii_next_leg_targets = front_lifting_targets 
                        self.left_iv_next_leg_targets = back_lifting_targets 
                        self.right_i_next_leg_targets = front_lifting_targets  
                        self.right_iii_next_leg_targets = back_lifting_targets

                    case self.LegCycleState.Planting_Lifting:
                        self.current_state = self.LegCycleState.Passing_Reaching

                        self.left_i_next_leg_targets = front_passing_targets  
                        self.left_iii_next_leg_targets = back_passing_targets
                        self.right_ii_next_leg_targets = front_passing_targets 
                        self.right_iv_next_leg_targets = back_passing_targets 

                        self.left_ii_next_leg_targets = front_reaching_targets 
                        self.left_iv_next_leg_targets = back_reaching_targets 
                        self.right_i_next_leg_targets = front_reaching_targets  
                        self.right_iii_next_leg_targets = back_reaching_targets
                        
                    case self.LegCycleState.Passing_Reaching:
                        self.current_state = self.LegCycleState.Lifting_Planting

                        self.left_i_next_leg_targets = front_lifting_targets  
                        self.left_iii_next_leg_targets = back_lifting_targets
                        self.right_ii_next_leg_targets = front_lifting_targets 
                        self.right_iv_next_leg_targets = back_lifting_targets 

                        self.left_ii_next_leg_targets = front_planting_targets 
                        self.left_iv_next_leg_targets = back_planting_targets 
                        self.right_i_next_leg_targets = front_planting_targets  
                        self.right_iii_next_leg_targets = back_planting_targets

        self.leg_cycle(leg_set.left_i_leg, self.left_i_leg_targets, self.left_i_next_leg_targets)
        self.leg_cycle(leg_set.left_ii_leg, self.left_ii_leg_targets, self.left_ii_next_leg_targets)
        self.leg_cycle(leg_set.left_iii_leg, self.left_iii_leg_targets, self.left_iii_next_leg_targets)
        self.leg_cycle(leg_set.left_iv_leg, self.left_iv_leg_targets, self.left_iv_next_leg_targets)

        self.leg_cycle(leg_set.right_i_leg, self.right_i_leg_targets, self.right_i_next_leg_targets)
        self.leg_cycle(leg_set.right_ii_leg, self.right_ii_leg_targets, self.right_ii_next_leg_targets)
        self.leg_cycle(leg_set.right_iii_leg, self.right_iii_leg_targets, self.right_iii_next_leg_targets)
        self.leg_cycle(leg_set.right_iv_leg, self.right_iv_leg_targets, self.right_iv_next_leg_targets)