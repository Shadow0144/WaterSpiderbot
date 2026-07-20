
import math
import numpy as np
import mujoco

from .spiderleg import Spiderleg

class Spiderbot:
    def __init__(self):
        self.path_to_xml = 'assets/models/spider_test.xml'
        self.load_model()

    def load_model(self):
        spec = mujoco.MjSpec.from_file(self.path_to_xml)

        # Cephalothorax connects to coxa [then trochanter] then femur [then patella] then tibia [then metatarsus] [then tarsus] then claws

        # Left side
        self.left_i_leg = Spiderleg(spec, "left_i", [0.7, 0.1, 0.1], [-0.175, 0.2, 0.0], [0, 0, 45], 1.00)
        self.left_ii_leg = Spiderleg(spec, "left_ii", [0.7, 0.1, 0.2], [-0.25, 0.075, 0.0], [0, 0, 65], 0.90)
        self.left_iii_leg = Spiderleg(spec, "left_iii", [0.7, 0.1, 0.3], [-0.25, -0.075, 0.0], [0, 0, 115], 0.75)
        self.left_iv_leg = Spiderleg(spec, "left_iv", [0.7, 0.1, 0.4], [-0.175, -0.2, 0.0], [0, 0, 135], 1.10)
        # Right side
        self.right_i_leg = Spiderleg(spec, "right_i", [0.1, 0.7, 0.1], [0.175, 0.2, 0.0], [0, 0, 315], 1.00)
        self.right_ii_leg = Spiderleg(spec, "right_ii", [0.1, 0.7, 0.2], [0.25, 0.075, 0.0], [0, 0, 285], 0.90)
        self.right_iii_leg = Spiderleg(spec, "right_iii", [0.1, 0.7, 0.3], [0.25, -0.075, 0.0], [0, 0, 245], 0.75)
        self.right_iv_leg = Spiderleg(spec, "right_iv", [0.1, 0.7, 0.4], [0.175, -0.2, 0.0], [0, 0, 225], 1.10)

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.left_i_leg.set_model_data(self.model, self.data)
        self.left_ii_leg.set_model_data(self.model, self.data)
        self.left_iii_leg.set_model_data(self.model, self.data)
        self.left_iv_leg.set_model_data(self.model, self.data)
        self.right_i_leg.set_model_data(self.model, self.data)
        self.right_ii_leg.set_model_data(self.model, self.data)
        self.right_iii_leg.set_model_data(self.model, self.data)
        self.right_iv_leg.set_model_data(self.model, self.data)

    def walk_forward(self, time):
        sin_phase = np.sin(time)
        cos_phase = np.cos(time)

        sin_half_phase = np.sin(time + (2.0*math.pi/3.0))
        cos_half_phase = np.cos(time + (2.0*math.pi/3.0))

        coxa_target_angle = math.radians(30)
        femur_target_angle = math.radians(45)
        tibia_target_angle = math.radians(5)

        # Left side
        self.left_i_leg.set_leg_targets(-coxa_target_angle * cos_phase, femur_target_angle * cos_half_phase, tibia_target_angle * cos_half_phase)
        self.left_ii_leg.set_leg_targets(-coxa_target_angle * sin_phase, femur_target_angle * sin_half_phase, tibia_target_angle * sin_half_phase)
        self.left_iii_leg.set_leg_targets(-coxa_target_angle * cos_phase, femur_target_angle * cos_half_phase, tibia_target_angle * cos_half_phase)
        self.left_iv_leg.set_leg_targets(-coxa_target_angle * sin_phase, femur_target_angle * sin_half_phase, tibia_target_angle * sin_half_phase)
        
        # Right back
        self.right_i_leg.set_leg_targets(coxa_target_angle * sin_phase, femur_target_angle * sin_half_phase, tibia_target_angle * sin_half_phase)
        self.right_ii_leg.set_leg_targets(coxa_target_angle * cos_phase, femur_target_angle * cos_half_phase,  tibia_target_angle * cos_half_phase)
        self.right_iii_leg.set_leg_targets(coxa_target_angle * sin_phase, femur_target_angle * sin_half_phase, tibia_target_angle * sin_half_phase)
        self.right_iv_leg.set_leg_targets(coxa_target_angle * cos_phase, femur_target_angle * cos_half_phase, tibia_target_angle * cos_half_phase)