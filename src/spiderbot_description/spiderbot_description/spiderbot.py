"""Constructs a description and model of a Spiderbot."""

import os

from ament_index_python.packages import get_package_share_directory

import mujoco

from .spiderleg import SpiderLegSet
from .util import draw_leg_space_in_mujoco
from .util import sample_reachable_leg_space


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
        self.leg_set = SpiderLegSet(self.spec)

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        self.leg_set.set_model_data(self.model, self.data)

    def walk_forward(self, delta_time):
        """Walk the Spiderbot forward using the locomotion module."""
        self.locomotion_module.walk_forward(delta_time, self.leg_set)

    def test_leg(self):
        """Test the leg moves to a position."""
        self.leg_set.left_i_leg.move_claw_to_cartesian([0, 10, 0])

    def visualize_leg_space(self):
        """Calculate and visualize the points a leg end can reach."""
        if not self.visualized:
            self.visualized = True
            draw_leg_space_in_mujoco(self.spec, self.leg_set.left_i_leg,
                                     sample_reachable_leg_space(
                                         self.leg_set.left_i_leg))

            self.model = self.spec.compile()
            self.data = mujoco.MjData(self.model)

            self.leg_set.set_model_data(self.model, self.data)

    def set_claw_targets(self, targets):
        """Set the target for every leg's claw in Cartesian space."""
        self.leg_set.set_claw_targets(targets)
