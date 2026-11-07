"""Deep Actor-Critic policy functionality."""

from .parallel_policy_trainer import ParallelPolicyTrainer
from .policy.deep_soft_actor_critics_policy import DeepSoftActorCriticsPolicy
from .series_policy_trainer import SeriesPolicyTrainer

__all__ = [
    'DeepSoftActorCriticsPolicy',
    'ParallelPolicyTrainer',
    'SeriesPolicyTrainer'
]
