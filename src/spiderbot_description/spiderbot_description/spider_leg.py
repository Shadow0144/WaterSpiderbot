"""Create a leg or set of legs."""

import mujoco


class SpiderLeg:
    """A single Spiderbot leg with 3 joints."""

    def __init__(self, spec, leg_id,
                 base_rgb, pos, euler,
                 segment_length, left_side):
        """Initialize the spider leg."""
        self.leg_id = leg_id
        self.base_rgb = base_rgb
        self.segment_length = segment_length

        self.damping = 0.01

        try:
            self.target = spec.worldbody.add_body(name=f'{self.leg_id}_target',
                                                  pos=[0, 0, 0], mocap=True)
            self.target.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                 size=[0.015, 0.0, 0.0],
                                 rgba=[self.base_rgb[0] + 0.5,
                                       self.base_rgb[1] + 0.5,
                                       self.base_rgb[2] + 0.5,
                                       0.75])

            cephalothorax = spec.body('cephalothorax')
            if not cephalothorax:
                raise ValueError('Could not find cephalothorax')

            if left_side:
                coxa_axis = [0, 1, 0]
            else:
                coxa_axis = [0, -1, 0]

            cephalothorax.add_site(name=f'{self.leg_id}_leg_base',
                                   pos=pos, euler=euler)

            coxa = cephalothorax.add_body(name=f'{self.leg_id}_coxa',
                                          pos=pos, euler=euler)
            coxa.childclass = 'coxa'
            coxa.add_joint(name=f'{self.leg_id}_cephalothorax_coxa_joint',
                           axis=coxa_axis)
            coxa.add_geom(
                rgba=[self.base_rgb[0], self.base_rgb[1], self.base_rgb[2], 1])

            femur = coxa.add_body(name=f'{self.leg_id}_femur',
                                  pos=[0, 0.04, 0], euler=[45, 0, 0])
            femur.childclass = 'femur'
            femur.add_joint(name=f'{self.leg_id}_coxa_femur_joint')
            femur.add_geom(
                rgba=[self.base_rgb[0] + 0.1,
                      self.base_rgb[1] + 0.1,
                      self.base_rgb[2], 1],
                fromto=([0.0, 0.0, 0.0, 0.0, 0.0, -self.segment_length]))

            tibia = femur.add_body(name=f'{self.leg_id}_tibia',
                                   pos=[0, 0, -self.segment_length],
                                   euler=[-45, 0, 0])
            tibia.childclass = 'tibia'
            tibia.add_joint(name=f'{self.leg_id}_femur_tibia_joint')
            tibia.add_geom(
                rgba=[self.base_rgb[0] + 0.2,
                      self.base_rgb[1] + 0.2,
                      self.base_rgb[2], 1],
                fromto=([0.0, 0.0, 0.0, 0.0, 0.0, -self.segment_length]))

            claw_length = 0.025
            claw = tibia.add_body(name=f'{self.leg_id}_claw',
                                  pos=[0, 0, -self.segment_length])
            claw.add_site(name=f'{self.leg_id}_claw_tip',
                          pos=[0, 0, -claw_length])
            claw.childclass = 'claw'
            claw.add_geom()

        except KeyError:
            print(f'Key error: {self.leg_id}')
            exit()

    def set_model_data(self, model, data):
        """Set the model data after the spec is compiled."""
        self.model = model
        self.data = data

        self.servo_coxa_actuator_id = self.model.actuator(
            'servo_' + self.leg_id + '_coxa_pitch').id
        self.servo_femur_actuator_id = self.model.actuator(
            'servo_' + self.leg_id + '_femur_pitch').id
        self.servo_tibia_actuator_id = self.model.actuator(
            'servo_' + self.leg_id + '_tibia_pitch').id

        coxa_joint_id = self.model.joint(
            f'{self.leg_id}_cephalothorax_coxa_joint').id
        femur_joint_id = self.model.joint(
            f'{self.leg_id}_coxa_femur_joint').id
        tibia_joint_id = self.model.joint(
            f'{self.leg_id}_femur_tibia_joint').id
        self.leg_joint_ids = [coxa_joint_id, femur_joint_id, tibia_joint_id]

        coxa_joint_dof = self.model.jnt_dofadr[coxa_joint_id]
        femur_joint_dof = self.model.jnt_dofadr[femur_joint_id]
        tibia_joint_dof = self.model.jnt_dofadr[tibia_joint_id]

        coxa_joint_qpos_adr = self.model.jnt_qposadr[coxa_joint_id]
        femur_joint_qpos_adr = self.model.jnt_qposadr[femur_joint_id]
        tibia_joint_qpos_adr = self.model.jnt_qposadr[tibia_joint_id]

        self.leg_dof_adrs = [coxa_joint_dof,
                             femur_joint_dof,
                             tibia_joint_dof]
        self.leg_qpos_adrs = [coxa_joint_qpos_adr,
                              femur_joint_qpos_adr,
                              tibia_joint_qpos_adr]

        self.coxa_body_id = self.model.body(f'{self.leg_id}_coxa').id

        self.leg_base_site_id = self.model.site(f'{self.leg_id}_leg_base').id
        self.claw_tip_site_id = self.model.site(f'{self.leg_id}_claw_tip').id

        target_mocap_body_id = self.model.body(f'{self.leg_id}_target').id
        self.target_mocap_id = self.model.body_mocapid[target_mocap_body_id]

    def get_qposes(self):
        """Return the qposes of all actuators."""
        return self.data.qpos[self.leg_qpos_adrs]
