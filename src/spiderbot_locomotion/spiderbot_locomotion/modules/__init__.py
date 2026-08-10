"""Spiderbot locomotion modules."""

from .deep_actor_critic_module import DeepActorCriticModule
from .handcrafted_angle_module import HandcraftedAngleModule
from .handcrafted_point_module import HandcraftedPointModule
from .simple_sin_module import SimpleSinModule

__all__ = [
    'DeepActorCriticModule',
    'HandcraftedAngleModule',
    'HandcraftedPointModule',
    'SimpleSinModule',
]
