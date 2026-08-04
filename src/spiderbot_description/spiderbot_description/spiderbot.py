"""Constructs a description and model of a Spiderbot."""

import os

from ament_index_python.packages import get_package_share_directory

import mujoco

from .spider_leg import SpiderLeg


class Spiderbot:
    """Main description of a Spiderbot robot."""

    def __init__(self):
        """Initialize a Spiderbot."""
        self.visualized = False
        self.path_to_xml = self.get_spider_model_path()
        self.load_model()

    def get_spider_model_path(self):
        """Get the path to the model file from the share directory."""
        share_dir = get_package_share_directory('spiderbot_description')
        model_path = os.path.join(share_dir, 'models', 'spider_test.xml')
        return model_path

    def load_model(self):
        """
        Load and finish the model and data.

        Loads the Spiderbot partial description from a file
        and adds the legs, completing the model and data.
        """
        self.spec = mujoco.MjSpec.from_file(self.path_to_xml)

        # Cephalothorax connects to coxa [then trochanter] then femur
        # [then patella] then tibia [then metatarsus] [then tarsus] then claws
        self.create_legs(self.spec)

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        for leg_name in self.leg_names:
            self.legs[leg_name].set_model_data(self.model, self.data)

    def set_claw_targets(self, targets):
        """Set the target for every leg's claw in Cartesian space."""
        for leg_name in self.leg_names:
            self.legs[leg_name].set_claw_target(targets[leg_name])

    def create_legs(self, spec, use_anatomical_lengths=True):
        """Create the legs of the Spiderbot."""
        self.base_segment_length = 0.25
        self.rest_angles = {'coxa': 0.0, 'femur': 0.0, 'tibia': 0.0}

        self.leg_names = ['l_i', 'l_ii', 'l_iii', 'l_iv',
                          'r_i', 'r_ii', 'r_iii', 'r_iv']

        self.segment_lengths = [1.0, 1.0, 1.0, 1.0]
        # Leg ratios:
        if use_anatomical_lengths:
            self.segment_lengths[0] = self.base_segment_length * 1.00
            self.segment_lengths[1] = self.base_segment_length * 0.90
            self.segment_lengths[2] = self.base_segment_length * 0.75
            self.segment_lengths[3] = self.base_segment_length * 1.10
        else:
            self.segment_lengths[0] = self.base_segment_length * 1.00
            self.segment_lengths[1] = self.base_segment_length * 1.00
            self.segment_lengths[2] = self.base_segment_length * 1.00
            self.segment_lengths[3] = self.base_segment_length * 1.00

        self.segment_lengths.extend(self.segment_lengths)  # Left + right

        # Leg segment ratios:
        # Femur: 1.0
        # [Patella: 0.4]
        # Tibia: 1.0
        # [Metatarsus: 1.0 / 1.05]
        # [Tarsus: 0.4]

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

        poses = [
            [-0.175, 0.2, 0.0],
            [-0.25, 0.075, 0.0],
            [-0.25, -0.075, 0.0],
            [-0.175, -0.2, 0.0],
            [0.175, 0.2, 0.0],
            [0.25, 0.075, 0.0],
            [0.25, -0.075, 0.0],
            [0.175, -0.2, 0.0]
        ]

        eulers = [
            [0, 0, 45],
            [0, 0, 65],
            [0, 0, 115],
            [0, 0, 135],
            [0, 0, 315],
            [0, 0, 295],
            [0, 0, 245],
            [0, 0, 225]
        ]

        self.legs = {}
        for i in range(0, 8):
            self.legs[self.leg_names[i]] = SpiderLeg(
                spec,
                self.leg_names[i],
                base_rgb[i],
                poses[i],
                eulers[i],
                self.segment_lengths[i],
                i < 4)
