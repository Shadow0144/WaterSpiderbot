"""Storage class for a transition of a recurrent neural network policy."""

from dataclasses import dataclass

import torch


@dataclass
class StepTransition:
    """Store training state information from a step."""

    state_t: torch.Tensor
    action_t: torch.Tensor
    log_probability_t: torch.Tensor
    hidden_state_t: torch.Tensor
    hidden_state_tp1: torch.Tensor
