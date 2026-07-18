
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
            self.servo_hip_actuator_id = self.model.actuator("servo_" + self.id + "_hip_pitch").id
            self.servo_upper_leg_actuator_id = self.model.actuator("servo_" + self.id + "_upper_leg_pitch").id
            self.servo_lower_leg_actuator_id = self.model.actuator("servo_" + self.id + "_lower_leg_pitch").id
        except KeyError:
            print(f"Key error")
            exit()

    def set_hip_target(self, target_angle_rad):
        self.data.ctrl[self.servo_hip_actuator_id] = target_angle_rad

    def set_upper_leg_target(self, target_angle_rad):
        self.data.ctrl[self.servo_upper_leg_actuator_id] = target_angle_rad

    def set_lower_leg_target(self, target_angle_rad):
        self.data.ctrl[self.servo_lower_leg_actuator_id] = target_angle_rad

    def set_leg_targets(self, hip_target_angle_rad, upper_leg_target_angle_rad, lower_leg_target_angle_rad):
        self.set_hip_target(hip_target_angle_rad)
        self.set_upper_leg_target(upper_leg_target_angle_rad)
        self.set_lower_leg_target(lower_leg_target_angle_rad)


class spiderbot:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.load_model()

    def load_model(self):
        # Left side
        self.left_back_leg = spiderleg(self.model, self.data, "left_back")
        self.left_center_back_leg = spiderleg(self.model, self.data, "left_center_back")
        self.left_center_front_leg = spiderleg(self.model, self.data, "left_center_front")
        self.left_front_leg = spiderleg(self.model, self.data, "left_front")
        # Right side
        self.right_back_leg = spiderleg(self.model, self.data, "right_back")
        self.right_center_back_leg = spiderleg(self.model, self.data, "right_center_back")
        self.right_center_front_leg = spiderleg(self.model, self.data, "right_center_front")
        self.right_front_leg = spiderleg(self.model, self.data, "right_front")

    def walk_forward(self, time):
        sin_phase = np.sin(time)
        cos_phase = np.cos(time)

        sin_half_phase = np.sin(time + (2.0*math.pi/3.0))
        cos_half_phase = np.cos(time + (2.0*math.pi/3.0))

        hip_target_angle = math.radians(30)
        upper_leg_target_angle = math.radians(45)
        lower_leg_target_angle = math.radians(5)

        # Left side
        self.left_back_leg.set_leg_targets(-hip_target_angle * sin_phase, upper_leg_target_angle * sin_half_phase, lower_leg_target_angle * sin_half_phase)
        self.left_center_back_leg.set_leg_targets(-hip_target_angle * cos_phase, upper_leg_target_angle * cos_half_phase, lower_leg_target_angle * cos_half_phase)
        self.left_center_front_leg.set_leg_targets(-hip_target_angle * sin_phase, upper_leg_target_angle * sin_half_phase, lower_leg_target_angle * sin_half_phase)
        self.left_front_leg.set_leg_targets(-hip_target_angle * cos_phase, upper_leg_target_angle * cos_half_phase, lower_leg_target_angle * cos_half_phase)
        
        # Right back
        self.right_back_leg.set_leg_targets(hip_target_angle * cos_phase, upper_leg_target_angle * cos_half_phase, lower_leg_target_angle * cos_half_phase)
        self.right_center_back_leg.set_leg_targets(hip_target_angle * sin_phase, upper_leg_target_angle * sin_half_phase, lower_leg_target_angle * sin_half_phase)
        self.right_center_front_leg.set_leg_targets(hip_target_angle * cos_phase, upper_leg_target_angle * cos_half_phase,  lower_leg_target_angle * cos_half_phase)
        self.right_front_leg.set_leg_targets(hip_target_angle * sin_phase, upper_leg_target_angle * sin_half_phase, lower_leg_target_angle * sin_half_phase)