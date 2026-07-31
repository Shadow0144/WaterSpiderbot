
import math
import numpy as np
import mujoco

from .spiderleg import SpiderLegSet
from .locomotion import SimpleSinLocomotionModule
from .locomotion import HandcraftedLocomotionModule
from .locomotion import MoveToPointLocomotionModule

from . import util

class Spiderbot:
    def __init__(self):
        self.visualized = False
        self.path_to_xml = 'assets/models/spider_test.xml'
        self.load_model()

    def load_model(self):
        self.spec = mujoco.MjSpec.from_file(self.path_to_xml)

        # Cephalothorax connects to coxa [then trochanter] then femur [then patella] then tibia [then metatarsus] [then tarsus] then claws
        self.leg_set = SpiderLegSet(self.spec)

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        self.leg_set.set_model_data(self.model, self.data)

        self.locomotion_module = MoveToPointLocomotionModule(self.leg_set)

    def walk_forward(self, delta_time):
        self.locomotion_module.walk_forward(delta_time, self.leg_set)

    def test_leg(self, delta_time):
        self.leg_set.left_i_leg.move_claw_to_cartesian([0, 10, 0])

    def visualize_leg_space(self):
        if not self.visualized:
            self.visualized = True
            util.draw_leg_space_in_mujoco(self.spec, self.leg_set.left_i_leg, util.sample_reachable_leg_space(self.leg_set.left_i_leg))

            self.model = self.spec.compile()
            self.data = mujoco.MjData(self.model)

            self.leg_set.set_model_data(self.model, self.data)