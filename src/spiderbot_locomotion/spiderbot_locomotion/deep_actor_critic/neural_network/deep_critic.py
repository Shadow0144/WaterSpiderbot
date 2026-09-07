"""Locomotion Critic using a Deep Neural Network."""

import torch
from torch import nn


class DeepCritic(nn.Module):
    """Locomotion Critic using a Deep Neural Network."""

    def __init__(
            self,
            num_inputs,
            num_hiddens,
            num_outputs):
        """Initialize the Critic."""
        super().__init__()

        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs

        self.critic = nn.Sequential(
            nn.Linear(self.num_inputs,
                      self.num_hiddens),
            nn.ReLU(),
            nn.Linear(self.num_hiddens,
                      self.num_outputs)
        )

    def forward(self, hidden_state_tp1, action_t):
        """Forward pass."""
        critic_input = torch.cat([hidden_state_tp1, action_t], dim=-1)

        critic_value_t = self.critic(critic_input)

        return critic_value_t
