
import math
import numpy as np
import mujoco

from .spiderleg import SpiderLegSet
from .locomotion import SimpleSinLocomotionModule
from .locomotion import HandcraftedLocomotionModule

class Spiderbot:
    def __init__(self):
        self.path_to_xml = 'assets/models/spider_test.xml'
        self.load_model()

    def load_model(self):
        spec = mujoco.MjSpec.from_file(self.path_to_xml)

        # Cephalothorax connects to coxa [then trochanter] then femur [then patella] then tibia [then metatarsus] [then tarsus] then claws
        self.leg_set = SpiderLegSet(spec)

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.leg_set.set_model_data(self.model, self.data)

        self.locomotion_module = HandcraftedLocomotionModule(self.leg_set)

    def walk_forward(self, delta_time):
        self.locomotion_module.walk_forward(delta_time, self.leg_set)