"""Spiderbot locomotion."""

from .handcrafted import HandcraftedLocomotionModule
from .move_to_point import MoveToPointLocomotionModule
from .simple_sin import SimpleSinLocomotionModule

__all__ = [
    'HandcraftedLocomotionModule',
    'MoveToPointLocomotionModule',
    'SimpleSinLocomotionModule',
]
