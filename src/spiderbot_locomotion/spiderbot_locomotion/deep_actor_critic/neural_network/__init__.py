"""Spiderbot neural network components."""

from .deep_actor import DeepActor
from .deep_critic import DeepCritic

__all__ = [
    'DeepActor',
    'DeepCritic',
]
