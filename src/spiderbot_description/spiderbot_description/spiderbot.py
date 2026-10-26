"""Constructs a description and model of a Spiderbot."""

import os

from ament_index_python.packages import get_package_share_directory

import mujoco


class Spiderbot:
    """Main description of a Spiderbot robot."""

    def __init__(self):
        """Initialize a Spiderbot."""
        self.visualized = False
        self._load_model()

    def get_texture_path(self):
        """Get the path to the texture directory from the share directory."""
        texture_path = os.path.join(self._get_model_path(), 'textures')
        return texture_path

    def get_mesh_path(self):
        """Get the path to the mesh directory from the share directory."""
        mesh_path = os.path.join(self._get_model_path(), 'meshes')
        return mesh_path

    def get_spec_xml(self):
        """Get the spec xml."""
        return self.spec.to_xml()

    def _get_model_path(self):
        """Get the path to the model directory from the share directory."""
        share_dir = get_package_share_directory('spiderbot_description')
        model_path = os.path.join(share_dir, 'models')
        return model_path

    def _get_spiderbot_model_file(self):
        """Get the file path to the Spiderbot model file."""
        model_filepath = os.path.join(
            self._get_model_path(),
            'spiderbot_base.xml'
        )
        return model_filepath

    def _load_model(self):
        """
        Load and finish the model and data.

        Loads the Spiderbot partial description from a file
        and adds the legs, completing the model and data.
        """
        self.spiderbot_xml = self._get_spiderbot_model_file()

        self.spec = mujoco.MjSpec.from_file(self.spiderbot_xml)

        # Cephalothorax connects to coxa [then trochanter] then femur
        # [then patella] then tibia [then metatarsus] [then tarsus] then claws
        self._create_legs(self.spec)

        # Fix the mesh and texture directory paths
        self.spec.compiler.meshdir = self.get_mesh_path()
        self.spec.compiler.texturedir = self.get_texture_path()

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

    def _create_legs(self, spec, use_anatomical_lengths=True):
        """Create the legs of the Spiderbot."""
        self.damping = 0.01
        self.rest_angles = {'coxa': 0.0, 'femur': 0.0, 'tibia': 0.0}

        self.leg_names = [
            'l_i', 'l_ii', 'l_iii', 'l_iv',
            'r_i', 'r_ii', 'r_iii', 'r_iv'
        ]

        # Leg segment ratios:
        # Coxa: 0.0
        # [Trochanter: 0.15]
        # Femur: 1.0
        # [Patella: 0.3]
        # Tibia: 1.0
        # [Metatarsus: 0.6]
        # [Tarsus: 0.4]
        self.segment_lengths_per_leg = []
        self.base_segment_lengths = [0.0, 0.15, 0.15, 0.5]
        # l/r_i - l/r_iv
        if use_anatomical_lengths:
            self.leg_scales = [1.00, 0.90, 0.75, 1.10]
        else:
            self.leg_scales = [1.00, 1.00, 1.00, 1.00]
        # Scale each leg segment per leg by each whole leg length
        for i, leg_scale in enumerate(self.leg_scales):
            segment_lengths = []
            for segment_length in self.base_segment_lengths:
                segment_lengths.append(segment_length * leg_scale)
            self.segment_lengths_per_leg.append(segment_lengths)
        # Left + right
        self.segment_lengths_per_leg.extend(self.segment_lengths_per_leg)

        base_rgb = [
            [0.7, 0.1, 0.1],
            [0.7, 0.1, 0.2],
            [0.7, 0.1, 0.3],
            [0.7, 0.1, 0.4],
            [0.1, 0.7, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.7, 0.3],
            [0.1, 0.7, 0.4]
        ]

        coxa_attachment_sites = [
            [0.180, 0.190, 0.000],
            [0.060, 0.240, 0.000],
            [-0.060, 0.240, 0.000],
            [-0.180, 0.190, 0.000],
            [0.180, -0.190, 0.000],
            [0.060, -0.240, 0.000],
            [-0.060, -0.240, 0.000],
            [-0.180, -0.190, 0.000]
        ]

        coxa_eulers = [
            [0, 0, -45],
            [0, 0, -15],
            [0, 0, 15],
            [0, 0, 45],
            [0, 0, -135],
            [0, 0, -165],
            [0, 0, 165],
            [0, 0, 135]
        ]

        for i, leg_name in enumerate(self.leg_names):
            self._build_leg(
                spec,
                leg_name,
                base_rgb[i],
                coxa_attachment_sites[i],
                coxa_eulers[i],
                self.segment_lengths_per_leg[i],
                i < 4
            )

    def _build_leg(
            self,
            spec,
            leg_id,
            base_rgb,
            coxa_attachment_site,
            coxa_euler,
            segment_lengths,
            is_left_side
            ):
        """Build a spider leg."""
        try:
            # Claw target
            claw_target = spec.worldbody.add_body(
                name=f'{leg_id}_target', pos=[0, 0, 0], mocap=True
            )
            claw_target.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.015],
                rgba=[
                    base_rgb[0] + 0.5,
                    base_rgb[1] + 0.5,
                    base_rgb[2] + 0.5,
                    0.75
                ],
                contype=0,
                conaffinity=0
            )

            cephalothorax = spec.body('cephalothorax')
            if not cephalothorax:
                raise ValueError('Could not find cephalothorax')

            # Coxa
            if is_left_side:
                coxa_axis = [0, 0, 1]
            else:
                coxa_axis = [0, 0, 1]

            cephalothorax.add_site(
                name=f'{leg_id}_leg_base',
                pos=coxa_attachment_site,
                euler=coxa_euler
            )

            coxa = cephalothorax.add_body(
                name=f'{leg_id}_coxa',
                pos=coxa_attachment_site,
                euler=coxa_euler
            )
            coxa.childclass = 'coxa'
            coxa.add_joint(
                name=f'{leg_id}_cephalothorax_coxa_joint', axis=coxa_axis
            )
            coxa.add_geom(
                rgba=[base_rgb[0], base_rgb[1], base_rgb[2], 1]
            )
            coxa.add_site(
                name=f'{leg_id}_coxa_entry', pos=[0.0, 0.0, 0.0]
            )
            coxa.add_site(
                name=f'{leg_id}_coxa_exit', pos=[0.0, 0.03, 0.0]
            )

            # Femur
            femur = coxa.add_body(
                name=f'{leg_id}_femur',
                pos=[0, 0.05, 0], euler=[180, 0, 0]
            )
            femur.childclass = 'femur'
            femur.add_joint(name=f'{leg_id}_coxa_femur_joint')
            femur.add_geom(
                rgba=[
                    base_rgb[0] + 0.1,
                    base_rgb[1] + 0.1,
                    base_rgb[2],
                    1
                ],
                fromto=([0.0, 0.0, 0.0, 0.0, 0.0, -segment_lengths[1]])
            )
            femur.add_site(
                name=f'{leg_id}_femur_flexor',
                pos=[0.0, 0.015, -segment_lengths[1] * 0.75]
            )

            # Tibia
            tibia = femur.add_body(
                name=f'{leg_id}_tibia',
                pos=[0, 0, -segment_lengths[1]],
                euler=[-45, 0, 0]
            )
            tibia.childclass = 'tibia'
            tibia.add_joint(name=f'{leg_id}_femur_tibia_joint')
            tibia.add_geom(
                rgba=[
                    base_rgb[0] + 0.2,
                    base_rgb[1] + 0.2,
                    base_rgb[2],
                    1
                ],
                fromto=([0.0, 0.0, 0.0, 0.0, 0.0, -segment_lengths[2]])
            )
            tibia.add_site(
                name=f'{leg_id}_tibia_flexor',
                pos=[0.0, -0.01, -segment_lengths[2] * 0.75]
            )

            # Tarsus
            tarsus = tibia.add_body(
                name=f'{leg_id}_tarsus',
                pos=[0, 0, -segment_lengths[2]],
                euler=[-90, 0, 0]
            )
            tarsus.childclass = 'tarsus'
            tarsus.add_joint(name=f'{leg_id}_tibia_tarsus_joint')
            tarsus.add_geom(
                rgba=[
                    base_rgb[0] + 0.3,
                    base_rgb[1] + 0.3,
                    base_rgb[2],
                    1
                ],
                fromto=([0.0, 0.0, 0.0, 0.0, 0.0, -segment_lengths[3]])
            )
            tarsus.add_site(
                name=f'{leg_id}_tarsus_flexor',
                pos=[0.0, -0.01, -segment_lengths[3] * 0.75]
            )

            # Claw
            claw_length = 0.025
            claw = tarsus.add_body(
                name=f'{leg_id}_claw',
                pos=[0, 0, -segment_lengths[3]]
            )
            claw.add_site(
                name=f'{leg_id}_claw_tip',
                pos=[0, 0, -claw_length]
            )
            claw.childclass = 'claw'
            claw.add_geom()

            # Tendon
            tendon_length = 0.25 + sum(segment_lengths)

            tendon = spec.add_tendon(
                name=f'{leg_id}_tendon',
                stiffness=20.0,
                springlength=[tendon_length, tendon_length],
                damping=0.2
            )

            tendon.wrap_site('tendon_tension_site')
            tendon.wrap_site('body_anchor_site')
            tendon.wrap_site(f'{leg_id}_coxa_entry')
            tendon.wrap_site(f'{leg_id}_coxa_exit')
            tendon.wrap_site(f'{leg_id}_femur_flexor')
            tendon.wrap_site(f'{leg_id}_tibia_flexor')
            tendon.wrap_site(f'{leg_id}_tarsus_flexor')

        except KeyError:
            print(f'Key error: {leg_id}')
            exit()
