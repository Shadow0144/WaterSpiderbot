"""Storage class for storing information on an training observation."""

import torch


class TrainingObservation:
    """Store training observation information."""

    def __init__(
            self,
            state_t: torch.Tensor,
            action_t: torch.Tensor,
            log_probability_t: torch.Tensor):
        """Initialize the training observation."""
        self.state_t = state_t
        self.action_t = action_t
        self.log_probability_t = log_probability_t

    def update(
        self,
        state_tp1: torch.Tensor,
        reward_t: float,
        episode_terminated: bool
    ):
        """Update with the results of the action."""
        self.state_tp1 = state_tp1
        self.reward_t = reward_t
        self.episode_terminated = episode_terminated
