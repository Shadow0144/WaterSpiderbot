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
                      self.num_hiddens),
            nn.ReLU(),
            nn.Linear(self.num_hiddens,
                      self.num_outputs)
        )

    def forward(self, state_t, action_t=None):
        """Forward pass."""
        if action_t is None:
            critic_input = state_t
        else:
            critic_input = torch.cat([state_t, action_t], dim=-1)

        critic_value_t = self.critic(critic_input)

        return critic_value_t
