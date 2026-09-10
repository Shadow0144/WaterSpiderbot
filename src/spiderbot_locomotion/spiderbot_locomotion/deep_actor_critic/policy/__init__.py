"""Spiderbot locomotion policies."""

from .deep_actor_critic_policy import DeepActorCriticPolicy
from .deep_soft_actor_critics_policy import DeepSoftActorCriticsPolicy
from .population_trainer import PopulationTrainer

__all__ = [
    'DeepActorCriticPolicy',
    'DeepSoftActorCriticsPolicy',
    'PopulationTrainer',
]
