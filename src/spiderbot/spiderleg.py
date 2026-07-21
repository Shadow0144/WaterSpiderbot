class SpiderLegSet:
    def __init__(self, spec, use_anatomical_lengths=True):
        # Leg ratios:
        if use_anatomical_lengths:
            leg_i_len = 1.00
            leg_ii_len = 0.90
            leg_iii_len = 0.75
            leg_iv_len = 1.10
        else:
            leg_i_len = 1.00
            leg_ii_len = 1.00
            leg_iii_len = 1.00
            leg_iv_len = 1.00

        # Leg segment ratios:
        # Femur: 1.0
        # [Patella: 0.4]
        # Tibia: 1.0
        # [Metatarsus: 1.0 / 1.05]
        # [Tarsus: 0.4]
    
        # Left side
        self.left_i_leg = SpiderLeg(spec, "left_i", [0.7, 0.1, 0.1], [-0.175, 0.2, 0.0], [0, 0, 45], leg_i_len, False)
        self.left_ii_leg = SpiderLeg(spec, "left_ii", [0.7, 0.1, 0.2], [-0.25, 0.075, 0.0], [0, 0, 65], leg_ii_len, False)
        self.left_iii_leg = SpiderLeg(spec, "left_iii", [0.7, 0.1, 0.3], [-0.25, -0.075, 0.0], [0, 0, 115], leg_iii_len, False)
        self.left_iv_leg = SpiderLeg(spec, "left_iv", [0.7, 0.1, 0.4], [-0.175, -0.2, 0.0], [0, 0, 135], leg_iv_len, False)
        # Right side
        self.right_i_leg = SpiderLeg(spec, "right_i", [0.1, 0.7, 0.1], [0.175, 0.2, 0.0], [0, 0, 315], leg_i_len, True)
        self.right_ii_leg = SpiderLeg(spec, "right_ii", [0.1, 0.7, 0.2], [0.25, 0.075, 0.0], [0, 0, 295], leg_ii_len, True)
        self.right_iii_leg = SpiderLeg(spec, "right_iii", [0.1, 0.7, 0.3], [0.25, -0.075, 0.0], [0, 0, 245], leg_iii_len, True)
        self.right_iv_leg = SpiderLeg(spec, "right_iv", [0.1, 0.7, 0.4], [0.175, -0.2, 0.0], [0, 0, 225], leg_iv_len, True)

    def set_model_data(self, model, data):
        self.model = model
        self.data = data
        
        self.left_i_leg.set_model_data(self.model, self.data)
        self.left_ii_leg.set_model_data(self.model, self.data)
        self.left_iii_leg.set_model_data(self.model, self.data)
        self.left_iv_leg.set_model_data(self.model, self.data)
        self.right_i_leg.set_model_data(self.model, self.data)
        self.right_ii_leg.set_model_data(self.model, self.data)
        self.right_iii_leg.set_model_data(self.model, self.data)
        self.right_iv_leg.set_model_data(self.model, self.data)


class SpiderLeg:
    def __init__(self, spec, id, base_rgb, pos, euler, length_ratio, mirror):
        self.id = id
        self.base_rgb = base_rgb

        try:
            cephalothorax = spec.body("cephalothorax")
            if not cephalothorax:
                raise ValueError("Could not find cephalothorax")
            
            if mirror:
                coxa_axis = [0, -1, 0]
            else:
                coxa_axis = [0, 1, 0]
                
            coxa = cephalothorax.add_body(name=f"{self.id}_coxa", pos=pos, euler=euler)
            coxa.childclass = "coxa"
            coxa.add_joint(name=f"{self.id}_cephalothorax_coxa_joint", axis=coxa_axis)
            coxa.add_geom(rgba=[self.base_rgb[0], self.base_rgb[1], self.base_rgb[2], 1])
            
            femur = coxa.add_body(name=f"{self.id}_femur", pos=[0, 0.04, 0], euler=[45, 0, 0])
            femur.childclass = "femur"
            femur.add_joint(name=f"{self.id}_coxa_femur_joint")
            femur.add_geom(rgba=[self.base_rgb[0] + 0.1, self.base_rgb[1] + 0.1, self.base_rgb[2], 1], fromto=([0.0, 0.0, 0.0, 0.0, 0.0, -0.25 * length_ratio]))

            tibia = femur.add_body(name=f"{self.id}_tibia", pos=[0, 0, -0.25 * length_ratio], euler=[-45, 0, 0])
            tibia.childclass = "tibia"
            tibia.add_joint(name=f"{self.id}_femur_tibia_joint")
            tibia.add_geom(rgba=[self.base_rgb[0] + 0.2, self.base_rgb[1] + 0.2, self.base_rgb[2], 1], fromto=([0.0, 0.0, 0.0, 0.0, 0.0, -0.25 * length_ratio]))

            claw = tibia.add_body(name=f"{self.id}_claw", pos=[0, 0, -0.25 * length_ratio])
            claw.childclass = "claw"
            claw.add_geom()

        except KeyError:
            print(f"Key error: " + self.id)
            exit()

    def set_model_data(self, model, data):
        self.model = model
        self.data = data

        self.servo_coxa_actuator_id = self.model.actuator("servo_" + self.id + "_coxa_pitch").id
        self.servo_femur_actuator_id = self.model.actuator("servo_" + self.id + "_femur_pitch").id
        self.servo_tibia_actuator_id = self.model.actuator("servo_" + self.id + "_tibia_pitch").id

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