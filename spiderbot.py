
import math
import numpy as np
import mujoco

class spiderleg:
    def __init__(self, spec, model, data, id):
        self.model = model
        self.data = data
        self.id = id
        self.load_leg(spec)

    # Leg ratios:
    # I: 1.00
    # II: 0.90
    # III: 0.75
    # IV: 1.10

    # Leg segment ratios:
    # Femur: 1.0
    # [Patella: 0.4]
    # Tibia: 1.0
    # [Metatarsus: 1.0 / 1.05]
    # [Tarsus: 0.4]

    def load_leg(self, spec):
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
    def __init__(self, path_to_xml):
        self.model = mujoco.MjModel.from_xml_path(path_to_xml)
        self.data = mujoco.MjData(self.model)
        self.load_model()

    def load_model(self):
        spec = mujoco.MjSpec()

        # Left side
        self.left_i_leg = spiderleg(spec, self.model, self.data, "left_i")
        self.left_ii_leg = spiderleg(spec, self.model, self.data, "left_ii")
        self.left_iii_leg = spiderleg(spec, self.model, self.data, "left_iii")
        self.left_iv_leg = spiderleg(spec, self.model, self.data, "left_iv")
        # Right side
        self.right_i_leg = spiderleg(spec, self.model, self.data, "right_i")
        self.right_ii_leg = spiderleg(spec, self.model, self.data, "right_ii")
        self.right_iii_leg = spiderleg(spec, self.model, self.data, "right_iii")
        self.right_iv_leg = spiderleg(spec, self.model, self.data, "right_iv")

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