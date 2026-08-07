"""Spiderbot locomotion modules."""

from .dnn_module import DNNModule
from .handcrafted_angle_module import HandcraftedAngleModule
from .handcrafted_point_module import HandcraftedPointModule
from .simple_sin_module import SimpleSinModule

__all__ = [
    'DNNModule',
    'HandcraftedAngleModule',
    'HandcraftedPointModule',
    'SimpleSinModule',
]
