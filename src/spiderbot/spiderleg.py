class Spiderleg:
    def __init__(self, spec, id, base_rgb, pos, euler, length_ratio):
        self.id = id
        self.base_rgb = base_rgb
        self.load_leg(spec, pos, euler, length_ratio)

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

    def load_leg(self, spec, pos, euler, length_ratio):
        try:
            cephalothorax = spec.body("cephalothorax")
            if not cephalothorax:
                raise ValueError("Could not find cephalothorax")
                
            coxa = cephalothorax.add_body(name=f"{self.id}_coxa", pos=pos, euler=euler)
            coxa.childclass = "coxa"
            coxa.add_joint(name=f"{self.id}_cephalothorax_coxa_joint")
            coxa.add_geom(rgba=[self.base_rgb[0], self.base_rgb[1], self.base_rgb[2], 1])
            
            femur = coxa.add_body(name=f"{self.id}_femur", pos=[0, 0.04, 0], euler=[45, 0, 0])
            femur.childclass = "femur"
            femur.add_joint(name=f"{self.id}_coxa_femur_joint")
            femur.add_geom(rgba=[self.base_rgb[0] + 0.1, self.base_rgb[1] + 0.1, self.base_rgb[2], 1], fromto=([0.0, 0.0, 0.0, 0.0, 0.0, -0.25 * length_ratio]))

            tibia = femur.add_body(name=f"{self.id}_tibia", pos=[0, 0, -0.25 * length_ratio], euler=[315, 0, 0])
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