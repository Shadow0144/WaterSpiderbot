import math
import numpy as np

from .locomotion import LocomotionModule

class SimpleSinLocomotionModule(LocomotionModule):
    def __init__(self, leg_set):
        pass

    def walk_forward(self, time, leg_set):
        sin_phase = np.sin(time)
        cos_phase = np.cos(time)

        sin_half_phase = np.sin(time + (2.0*math.pi/3.0))
        cos_half_phase = np.cos(time + (2.0*math.pi/3.0))

        coxa_target_angle = math.radians(-30)
        femur_target_angle = math.radians(45)
        tibia_target_angle = math.radians(5)

        # Left side
        leg_set.left_i_leg.set_leg_targets(coxa_target_angle * cos_phase, femur_target_angle * cos_half_phase, tibia_target_angle * cos_half_phase)
        leg_set.left_ii_leg.set_leg_targets(coxa_target_angle * sin_phase, femur_target_angle * sin_half_phase, tibia_target_angle * sin_half_phase)
        leg_set.left_iii_leg.set_leg_targets(coxa_target_angle * cos_phase, femur_target_angle * cos_half_phase, tibia_target_angle * cos_half_phase)
        leg_set.left_iv_leg.set_leg_targets(coxa_target_angle * sin_phase, femur_target_angle * sin_half_phase, tibia_target_angle * sin_half_phase)
        
        # Right side
        leg_set.right_i_leg.set_leg_targets(coxa_target_angle * sin_phase, femur_target_angle * sin_half_phase, tibia_target_angle * sin_half_phase)
        leg_set.right_ii_leg.set_leg_targets(coxa_target_angle * cos_phase, femur_target_angle * cos_half_phase,  tibia_target_angle * cos_half_phase)
        leg_set.right_iii_leg.set_leg_targets(coxa_target_angle * sin_phase, femur_target_angle * sin_half_phase, tibia_target_angle * sin_half_phase)
        leg_set.right_iv_leg.set_leg_targets(coxa_target_angle * cos_phase, femur_target_angle * cos_half_phase, tibia_target_angle * cos_half_phase)