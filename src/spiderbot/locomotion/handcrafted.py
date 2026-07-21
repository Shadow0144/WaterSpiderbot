import math
import numpy as np
from enum import Enum, auto

from .locomotion import LocomotionModule

phase_timer = 4500

front_lifting_targets = [math.radians(20), math.radians(30), math.radians(15)]
front_reaching_targets = [math.radians(-20), math.radians(30), math.radians(15)]
front_planting_targets = [math.radians(-20), math.radians(0), math.radians(0)]
front_passing_targets = [math.radians(20), math.radians(0), math.radians(0)]

back_lifting_targets = [math.radians(15), math.radians(30), math.radians(15)]
back_reaching_targets = [math.radians(-30), math.radians(30), math.radians(15)]
back_planting_targets = [math.radians(-30), math.radians(0), math.radians(0)]
back_passing_targets = [math.radians(15), math.radians(0), math.radians(0)]

leg_i_offsets = [math.radians(0), math.radians(60), math.radians(0)]
leg_ii_offsets = [math.radians(0), math.radians(15), math.radians(0)]
leg_iii_offsets = [math.radians(0), math.radians(10), math.radians(-10)]
leg_iv_offsets = [math.radians(0), math.radians(30), math.radians(-30)]

class HandcraftedLocomotionModule(LocomotionModule):

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

        self.left_i_leg_targets = leg_i_offsets
        self.left_i_next_leg_targets = front_lifting_targets + leg_i_offsets
        self.left_iii_leg_targets = leg_iii_offsets
        self.left_iii_next_leg_targets = back_lifting_targets + leg_iii_offsets
        
        self.right_ii_leg_targets = leg_ii_offsets
        self.right_ii_next_leg_targets = front_lifting_targets + leg_ii_offsets
        self.right_iv_leg_targets = leg_iv_offsets
        self.right_iv_next_leg_targets = back_lifting_targets + leg_iv_offsets

        # Group 2 (lii, liv, ri, riii)

        self.left_ii_leg_targets = leg_ii_offsets
        self.left_ii_next_leg_targets = front_passing_targets + leg_ii_offsets
        self.left_iv_leg_targets = leg_iv_offsets
        self.left_iv_next_leg_targets = back_passing_targets + leg_iv_offsets

        self.right_i_leg_targets = leg_i_offsets
        self.right_i_next_leg_targets = front_passing_targets + leg_i_offsets
        self.right_iii_leg_targets = leg_iii_offsets
        self.right_iii_next_leg_targets = back_passing_targets + leg_iii_offsets

        self.leg_cycle(leg_set.left_i_leg, leg_i_offsets, leg_i_offsets)
        self.leg_cycle(leg_set.left_ii_leg, leg_ii_offsets, leg_ii_offsets)
        self.leg_cycle(leg_set.left_iii_leg, leg_iii_offsets, leg_iii_offsets)
        self.leg_cycle(leg_set.left_iv_leg, leg_iv_offsets, leg_iv_offsets)

        self.leg_cycle(leg_set.right_i_leg, leg_i_offsets, leg_i_offsets)
        self.leg_cycle(leg_set.right_ii_leg, leg_ii_offsets, leg_ii_offsets)
        self.leg_cycle(leg_set.right_iii_leg, leg_iii_offsets, leg_iii_offsets)
        self.leg_cycle(leg_set.right_iv_leg, leg_iv_offsets, leg_iv_offsets)

    def leg_cycle(self, leg, leg_targets, next_leg_targets):
        percentage = self.timer / self.time_to_complete
        coxa_target_angle = ((leg_targets[0] * percentage) + (next_leg_targets[0] * (1.0 - percentage)))
        femur_target_angle = (leg_targets[1] * percentage) + (next_leg_targets[1] * (1.0 - percentage))
        tibia_target_angle = (leg_targets[2] * percentage) + (next_leg_targets[2] * (1.0 - percentage))
        leg.set_leg_targets(coxa_target_angle, femur_target_angle, tibia_target_angle)

    def walk_forward(self, delta_time, leg_set):
        self.timer -= delta_time

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

                        self.left_i_next_leg_targets = front_reaching_targets + leg_i_offsets  
                        self.left_iii_next_leg_targets = back_reaching_targets + leg_iii_offsets
                        self.right_ii_next_leg_targets = front_reaching_targets + leg_ii_offsets 
                        self.right_iv_next_leg_targets = back_reaching_targets + leg_iv_offsets 

                        self.left_ii_next_leg_targets = front_passing_targets + leg_ii_offsets 
                        self.left_iv_next_leg_targets = back_passing_targets + leg_iv_offsets 
                        self.right_i_next_leg_targets = front_passing_targets + leg_i_offsets  
                        self.right_iii_next_leg_targets = back_passing_targets + leg_iii_offsets

                    case self.LegCycleState.Reaching_Passing:
                        self.current_state = self.LegCycleState.Planting_Lifting

                        self.left_i_next_leg_targets = front_planting_targets + leg_i_offsets  
                        self.left_iii_next_leg_targets = back_planting_targets + leg_iii_offsets
                        self.right_ii_next_leg_targets = front_planting_targets + leg_ii_offsets 
                        self.right_iv_next_leg_targets = back_planting_targets + leg_iv_offsets 
                        
                        self.left_ii_next_leg_targets = front_lifting_targets + leg_ii_offsets 
                        self.left_iv_next_leg_targets = back_lifting_targets + leg_iv_offsets 
                        self.right_i_next_leg_targets = front_lifting_targets + leg_i_offsets  
                        self.right_iii_next_leg_targets = back_lifting_targets + leg_iii_offsets

                    case self.LegCycleState.Planting_Lifting:
                        self.current_state = self.LegCycleState.Passing_Reaching

                        self.left_i_next_leg_targets = front_passing_targets + leg_i_offsets  
                        self.left_iii_next_leg_targets = back_passing_targets + leg_iii_offsets
                        self.right_ii_next_leg_targets = front_passing_targets + leg_ii_offsets 
                        self.right_iv_next_leg_targets = back_passing_targets + leg_iv_offsets 

                        self.left_ii_next_leg_targets = front_reaching_targets + leg_ii_offsets 
                        self.left_iv_next_leg_targets = back_reaching_targets + leg_iv_offsets 
                        self.right_i_next_leg_targets = front_reaching_targets + leg_i_offsets  
                        self.right_iii_next_leg_targets = back_reaching_targets + leg_iii_offsets
                        
                    case self.LegCycleState.Passing_Reaching:
                        self.current_state = self.LegCycleState.Lifting_Planting

                        self.left_i_next_leg_targets = front_lifting_targets + leg_i_offsets  
                        self.left_iii_next_leg_targets = back_lifting_targets + leg_iii_offsets
                        self.right_ii_next_leg_targets = front_lifting_targets + leg_ii_offsets 
                        self.right_iv_next_leg_targets = back_lifting_targets + leg_iv_offsets 

                        self.left_ii_next_leg_targets = front_planting_targets + leg_ii_offsets 
                        self.left_iv_next_leg_targets = back_planting_targets + leg_iv_offsets 
                        self.right_i_next_leg_targets = front_planting_targets + leg_i_offsets  
                        self.right_iii_next_leg_targets = back_planting_targets + leg_iii_offsets

        self.leg_cycle(leg_set.left_i_leg, self.left_i_leg_targets, self.left_i_next_leg_targets)
        self.leg_cycle(leg_set.left_ii_leg, self.left_ii_leg_targets, self.left_ii_next_leg_targets)
        self.leg_cycle(leg_set.left_iii_leg, self.left_iii_leg_targets, self.left_iii_next_leg_targets)
        self.leg_cycle(leg_set.left_iv_leg, self.left_iv_leg_targets, self.left_iv_next_leg_targets)

        self.leg_cycle(leg_set.right_i_leg, self.right_i_leg_targets, self.right_i_next_leg_targets)
        self.leg_cycle(leg_set.right_ii_leg, self.right_ii_leg_targets, self.right_ii_next_leg_targets)
        self.leg_cycle(leg_set.right_iii_leg, self.right_iii_leg_targets, self.right_iii_next_leg_targets)
        self.leg_cycle(leg_set.right_iv_leg, self.right_iv_leg_targets, self.right_iv_next_leg_targets)