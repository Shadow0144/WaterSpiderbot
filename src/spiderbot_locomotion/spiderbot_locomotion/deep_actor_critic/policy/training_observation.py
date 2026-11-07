"""Storage class for storing information on an training observation."""

from dataclasses import dataclass

import torch


@dataclass
class TrainingObservation:
    """Store training observation information."""

    state_t: torch.Tensor
    action_t: torch.Tensor
    log_probability_t: torch.Tensor
    state_tp1: torch.Tensor
    reward_t: float
    episode_terminated: bool
