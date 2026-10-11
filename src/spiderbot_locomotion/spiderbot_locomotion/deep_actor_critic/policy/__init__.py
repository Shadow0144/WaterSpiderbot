"""Spiderbot locomotion policies."""

from .deep_actor_critic_policy import DeepActorCriticPolicy
from .deep_soft_actor_critics_policy import DeepSoftActorCriticsPolicy
from .parallel_policy_trainer import ParallelPolicyTrainer
from .series_policy_trainer import SeriesPolicyTrainer

__all__ = [
    'DeepActorCriticPolicy',
    'DeepSoftActorCriticsPolicy',
    'ParallelPolicyTrainer',
    'SeriesPolicyTrainer'
]
