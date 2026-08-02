"""Create a leg or set of legs."""

import mujoco

import numpy as np


class SpiderLegSet:
    """A set of 8 Spiderlegs."""

    def __init__(self, spec, use_anatomical_lengths=True):
        """Initialize a set of Spiderlegs."""
        self.base_leg_length = 0.25
        self.rest_angles = {'coxa': 0.0, 'femur': 0.0, 'tibia': 0.0}

        # Leg ratios:
        if use_anatomical_lengths:
            leg_i_len = self.base_leg_length * 1.00
            leg_ii_len = self.base_leg_length * 0.90
            leg_iii_len = self.base_leg_length * 0.75
            leg_iv_len = self.base_leg_length * 1.10
        else:
            leg_i_len = self.base_leg_length * 1.00
            leg_ii_len = self.base_leg_length * 1.00
            leg_iii_len = self.base_leg_length * 1.00
            leg_iv_len = self.base_leg_length * 1.00

        # Leg segment ratios:
        # Femur: 1.0
        # [Patella: 0.4]
        # Tibia: 1.0
        # [Metatarsus: 1.0 / 1.05]
        # [Tarsus: 0.4]

        # Left side
        self.left_i_leg = SpiderLeg(spec, 'left_i', [0.7, 0.1, 0.1],
                                    [-0.175, 0.2, 0.0], [0, 0, 45],
                                    leg_i_len, True)
        self.left_ii_leg = SpiderLeg(spec, 'left_ii', [0.7, 0.1, 0.2],
                                     [-0.25, 0.075, 0.0], [0, 0, 65],
                                     leg_ii_len, True)
        self.left_iii_leg = SpiderLeg(spec, 'left_iii', [0.7, 0.1, 0.3],
                                      [-0.25, -0.075, 0.0], [0, 0, 115],
                                      leg_iii_len, True)
        self.left_iv_leg = SpiderLeg(spec, 'left_iv', [0.7, 0.1, 0.4],
                                     [-0.175, -0.2, 0.0], [0, 0, 135],
                                     leg_iv_len, True)
        # Right side
        self.right_i_leg = SpiderLeg(spec, 'right_i', [0.1, 0.7, 0.1],
                                     [0.175, 0.2, 0.0], [0, 0, 315],
                                     leg_i_len, False)
        self.right_ii_leg = SpiderLeg(spec, 'right_ii', [0.1, 0.7, 0.2],
                                      [0.25, 0.075, 0.0], [0, 0, 295],
                                      leg_ii_len, False)
        self.right_iii_leg = SpiderLeg(spec, 'right_iii', [0.1, 0.7, 0.3],
                                       [0.25, -0.075, 0.0], [0, 0, 245],
                                       leg_iii_len, False)
        self.right_iv_leg = SpiderLeg(spec, 'right_iv', [0.1, 0.7, 0.4],
                                      [0.175, -0.2, 0.0], [0, 0, 225],
                                      leg_iv_len, False)

    def set_model_data(self, model, data):
        """Set the model data after the spec is compiled."""
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
    """A single spider's leg with 3 joints."""

    def __init__(self, spec, leg_id,
                 base_rgb, pos, euler,
                 leg_length, left_side):
        """Initialize the Spiderleg."""
        self.leg_id = leg_id
        self.base_rgb = base_rgb
        self.leg_length = leg_length

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
                fromto=([0.0, 0.0, 0.0, 0.0, 0.0, -leg_length]))

            tibia = femur.add_body(name=f'{self.leg_id}_tibia',
                                   pos=[0, 0, -leg_length],
                                   euler=[-45, 0, 0])
            tibia.childclass = 'tibia'
            tibia.add_joint(name=f'{self.leg_id}_femur_tibia_joint')
            tibia.add_geom(
                rgba=[self.base_rgb[0] + 0.2,
                      self.base_rgb[1] + 0.2,
                      self.base_rgb[2], 1],
                fromto=([0.0, 0.0, 0.0, 0.0, 0.0, -leg_length]))

            claw_length = 0.025
            claw = tibia.add_body(name=f'{self.leg_id}_claw',
                                  pos=[0, 0, -leg_length])
            claw.add_site(name=f'{self.leg_id}_claw_tip', pos=[0, 0, -claw_length])
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

        coxa_joint_qpos = self.model.jnt_qposadr[coxa_joint_id]
        femur_joint_qpos = self.model.jnt_qposadr[femur_joint_id]
        tibia_joint_qpos = self.model.jnt_qposadr[tibia_joint_id]

        self.leg_dof_adrs = [coxa_joint_dof,
                             femur_joint_dof,
                             tibia_joint_dof]
        self.leg_qpos_adrs = [coxa_joint_qpos,
                              femur_joint_qpos,
                              tibia_joint_qpos]

        self.coxa_body_id = self.model.body(f'{self.leg_id}_coxa').id

        self.leg_base_site_id = self.model.site(f'{self.leg_id}_leg_base').id
        self.claw_tip_site_id = self.model.site(f'{self.leg_id}_claw_tip').id

        target_mocap_body_id = self.model.body(f'{self.leg_id}_target').id
        self.target_mocap_id = self.model.body_mocapid[target_mocap_body_id]

    def set_coxa_target(self, target_angle_rad):
        """Set the target angle for the coxa joint."""
        self.data.ctrl[self.servo_coxa_actuator_id] = target_angle_rad

    def set_femur_target(self, target_angle_rad):
        """Set the target angle for the femur joint."""
        self.data.ctrl[self.servo_femur_actuator_id] = target_angle_rad

    def set_tibia_target(self, target_angle_rad):
        """Set the target angle for the tibia joint."""
        self.data.ctrl[self.servo_tibia_actuator_id] = target_angle_rad

    def set_leg_targets(self, coxa_target_angle_rad,
                        femur_target_angle_rad,
                        tibia_target_angle_rad):
        """Set the target angles for the leg joints."""
        self.set_coxa_target(coxa_target_angle_rad)
        self.set_femur_target(femur_target_angle_rad)
        self.set_tibia_target(tibia_target_angle_rad)

    def set_rest_targets(self, coxa_target_angle_rad,
                         femur_target_angle_rad,
                         tibia_target_angle_rad):
        """Set the angles for the leg joints to return to when resting."""
        self.rest_angles = {'coxa': coxa_target_angle_rad,
                            'femur': femur_target_angle_rad,
                            'tibia': tibia_target_angle_rad}

    def return_to_rest(self):
        """Move the leg joints to their rest positions."""
        self.set_leg_targets(self.rest_angles['coxa'],
                             self.rest_angles['femur'],
                             self.rest_angles['tibia'])

    # target_xyz is a vector of the target x, y, and z position
    # target_rpy is desired the roll, pitch, and yaw of the end effector
    def move_claw_to_cartesian(self, target_xyz, target_rpy=[], relative=True):
        """Move the tip of a leg to a point in Cartesian space."""
        # q_target = q_current + J^-1_p(x_target - x_current)
        # J is the Jacobian
        # q is the angles
        # x is the positions

        current_xyz = self.data.site(self.claw_tip_site_id).xpos

        # Get the Jacobian and then filter for the actuators we care about
        jac_p = np.zeros((3, self.model.nv))
        jac_r = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data,
                          jac_p, jac_r, self.claw_tip_site_id)
        j_leg = jac_p[:, self.leg_dof_adrs]

        if relative:
            r_leg_base = (
                self.data.site(self.leg_base_site_id).xmat.reshape(3, 3)
            )
            leg_base_xyz = self.data.site(self.leg_base_site_id).xpos
            current_xyz = r_leg_base.T @ (current_xyz - leg_base_xyz)
            j_leg = r_leg_base.T @ j_leg
            self.data.mocap_pos[self.target_mocap_id] = (
                leg_base_xyz + (r_leg_base @ target_xyz)
            )
        else:
            self.data.mocap_pos[self.target_mocap_id] = target_xyz

        # Damped pseudo-inverse matrix
        j_jt = j_leg @ j_leg.T + (self.damping**2) * np.eye(3)
        j_pinv = j_leg.T @ np.linalg.inv(j_jt)

        # Change in angles
        dxyz = (target_xyz - current_xyz).T
        d_q = j_pinv @ dxyz
        q_target = self.data.qpos[self.leg_qpos_adrs] + d_q

        coxa_target = q_target[0]
        femur_target = q_target[1]
        tibia_target = q_target[2]

        self.set_leg_targets(coxa_target, femur_target, tibia_target)
