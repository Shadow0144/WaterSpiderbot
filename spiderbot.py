
import math
import numpy as np
#import mujoco

class spiderleg:
    def __init__(self, model, data, id):
        self.model = model
        self.data = data
        self.id = id
        self.load_leg()

    def load_leg(self):
        try:
            self.servo_coxa_actuator_id = self.model.actuator("servo_" + self.id + "_coxa_pitch").id
            self.servo_femur_actuator_id = self.model.actuator("servo_" + self.id + "_femur_pitch").id
            self.servo_tibia_actuator_id = self.model.actuator("servo_" + self.id + "_tibia_pitch").id
        except KeyError:
            print(f"Key error: " + self.id)
            exit()

    def set_coxa_target(self, target_angle_rad):
        self.data.ctrl[self.servo_coxa_actuator_id] = target_angle_rad

    def set_femur_target(self, target_angle_rad):
        self.data.ctrl[self.servo_femur_actuator_id] = target_angle_rad

    def set_tibia_target(self, target_angle_rad):
        self.data.ctrl[self.servo_tibia_actuator_id] = target_angle_rad

    def set_leg_targets(self, coxa_target_angle_rad, femur_target_angle_rad, tibia_target_angle_rad):
        self.set_coxa_target(coxa_target_angle_rad)
        self.set_femur_target(femur_target_angle_rad)
        self.set_tibia_target(tibia_target_angle_rad)


class spiderbot:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.load_model()

    def load_model(self):
        # Left side
        self.left_i_leg = spiderleg(self.model, self.data, "left_i")
        self.left_ii_leg = spiderleg(self.model, self.data, "left_ii")
        self.left_iii_leg = spiderleg(self.model, self.data, "left_iii")
        self.left_iv_leg = spiderleg(self.model, self.data, "left_iv")
        # Right side
        self.right_i_leg = spiderleg(self.model, self.data, "right_i")
        self.right_ii_leg = spiderleg(self.model, self.data, "right_ii")
        self.right_iii_leg = spiderleg(self.model, self.data, "right_iii")
        self.right_iv_leg = spiderleg(self.model, self.data, "right_iv")

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