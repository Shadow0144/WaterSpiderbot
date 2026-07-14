
import math
import numpy as np
import mujoco

class spiderbot:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.load_model()

    def load_model(self):
        try:
            # Left hips
            self.servo_left_back_hip_actuator_id = self.model.actuator("servo_left_back_hip_pitch").id
            self.servo_left_center_back_hip_actuator_id = self.model.actuator("servo_left_center_back_hip_pitch").id
            self.servo_left_center_front_hip_actuator_id = self.model.actuator("servo_left_center_front_hip_pitch").id
            self.servo_left_front_hip_actuator_id = self.model.actuator("servo_left_front_hip_pitch").id
            
            # Left upper legs
            self.servo_left_back_upper_leg_actuator_id = self.model.actuator("servo_left_back_upper_leg_pitch").id
            self.servo_left_center_back_upper_leg_actuator_id = self.model.actuator("servo_left_center_back_upper_leg_pitch").id
            self.servo_left_center_front_upper_leg_actuator_id = self.model.actuator("servo_left_center_front_upper_leg_pitch").id
            self.servo_left_front_upper_leg_actuator_id = self.model.actuator("servo_left_front_upper_leg_pitch").id
            
            # Left lower legs
            self.servo_left_back_lower_leg_actuator_id = self.model.actuator("servo_left_back_lower_leg_pitch").id
            self.servo_left_center_back_lower_leg_actuator_id = self.model.actuator("servo_left_center_back_lower_leg_pitch").id
            self.servo_left_center_front_lower_leg_actuator_id = self.model.actuator("servo_left_center_front_lower_leg_pitch").id
            self.servo_left_front_lower_leg_actuator_id = self.model.actuator("servo_left_front_lower_leg_pitch").id

            # Right hips
            self.servo_right_back_hip_actuator_id = self.model.actuator("servo_right_back_hip_pitch").id
            self.servo_right_center_back_hip_actuator_id = self.model.actuator("servo_right_center_back_hip_pitch").id
            self.servo_right_center_front_hip_actuator_id = self.model.actuator("servo_right_center_front_hip_pitch").id
            self.servo_right_front_hip_actuator_id = self.model.actuator("servo_right_front_hip_pitch").id

            # Right upper legs
            self.servo_right_back_upper_leg_actuator_id = self.model.actuator("servo_right_back_upper_leg_pitch").id
            self.servo_right_center_back_upper_leg_actuator_id = self.model.actuator("servo_right_center_back_upper_leg_pitch").id
            self.servo_right_center_front_upper_leg_actuator_id = self.model.actuator("servo_right_center_front_upper_leg_pitch").id
            self.servo_right_front_upper_leg_actuator_id = self.model.actuator("servo_right_front_upper_leg_pitch").id

            # Right lower legs
            self.servo_right_back_lower_leg_actuator_id = self.model.actuator("servo_right_back_lower_leg_pitch").id
            self.servo_right_center_back_lower_leg_actuator_id = self.model.actuator("servo_right_center_back_lower_leg_pitch").id
            self.servo_right_center_front_lower_leg_actuator_id = self.model.actuator("servo_right_center_front_lower_leg_pitch").id
            self.servo_right_front_lower_leg_actuator_id = self.model.actuator("servo_right_front_lower_leg_pitch").id
        except KeyError:
            print(f"Key error")
            exit()

    def walk_forward(self, time):
        sin_phase = np.sin(time)
        cos_phase = np.cos(time)

        sin_half_phase = np.sin(time + (2.0*math.pi/3.0))
        cos_half_phase = np.cos(time + (2.0*math.pi/3.0))

        hip_target_angle = math.radians(30)
        upper_leg_target_angle = math.radians(45)
        lower_leg_target_angle = math.radians(5)

        # Left back
        self.data.ctrl[self.servo_left_back_hip_actuator_id] = -hip_target_angle * sin_phase
        self.data.ctrl[self.servo_left_back_upper_leg_actuator_id] = upper_leg_target_angle * sin_half_phase
        self.data.ctrl[self.servo_left_back_lower_leg_actuator_id] = lower_leg_target_angle * sin_half_phase

        # Left center back
        self.data.ctrl[self.servo_left_center_back_hip_actuator_id] = -hip_target_angle * cos_phase
        self.data.ctrl[self.servo_left_center_back_upper_leg_actuator_id] = upper_leg_target_angle * cos_half_phase
        self.data.ctrl[self.servo_left_center_back_lower_leg_actuator_id] = lower_leg_target_angle * cos_half_phase

        # Left center front
        self.data.ctrl[self.servo_left_center_front_hip_actuator_id] = -hip_target_angle * sin_phase
        self.data.ctrl[self.servo_left_center_front_upper_leg_actuator_id] = upper_leg_target_angle * sin_half_phase
        self.data.ctrl[self.servo_left_center_front_lower_leg_actuator_id] = lower_leg_target_angle * sin_half_phase

        # Left front
        self.data.ctrl[self.servo_left_front_hip_actuator_id] = -hip_target_angle * cos_phase
        self.data.ctrl[self.servo_left_front_upper_leg_actuator_id] = upper_leg_target_angle * cos_half_phase
        self.data.ctrl[self.servo_left_front_lower_leg_actuator_id] = lower_leg_target_angle * cos_half_phase
        
        # Right back
        self.data.ctrl[self.servo_right_back_hip_actuator_id] = hip_target_angle * cos_phase
        self.data.ctrl[self.servo_right_back_upper_leg_actuator_id] = upper_leg_target_angle * cos_half_phase
        self.data.ctrl[self.servo_right_back_lower_leg_actuator_id] = lower_leg_target_angle * cos_half_phase

        # Right center back
        self.data.ctrl[self.servo_right_center_back_hip_actuator_id] = hip_target_angle * sin_phase
        self.data.ctrl[self.servo_right_center_back_upper_leg_actuator_id] = upper_leg_target_angle * sin_half_phase
        self.data.ctrl[self.servo_right_center_back_lower_leg_actuator_id] = lower_leg_target_angle * sin_half_phase

        # Right center front
        self.data.ctrl[self.servo_right_center_front_hip_actuator_id] = hip_target_angle * cos_phase
        self.data.ctrl[self.servo_right_center_front_upper_leg_actuator_id] = upper_leg_target_angle * cos_half_phase
        self.data.ctrl[self.servo_right_center_front_lower_leg_actuator_id] = lower_leg_target_angle * cos_half_phase

        # Right front
        self.data.ctrl[self.servo_right_front_hip_actuator_id] = hip_target_angle * sin_phase
        self.data.ctrl[self.servo_right_front_upper_leg_actuator_id] = upper_leg_target_angle * sin_half_phase
        self.data.ctrl[self.servo_right_front_lower_leg_actuator_id] = lower_leg_target_angle * sin_half_phase