"""Locomotion Soft Critic using a Deep Neural Network."""

import torch
from torch import nn


class DeepSoftCritics(nn.Module):
    """Locomotion Soft Critics using a Deep Neural Network."""

    def __init__(
            self,
            num_inputs,
            num_hiddens,
            num_outputs):
        """Initialize the Critics."""
        super().__init__()

        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs

        # Two critics
        self.critic_q1 = nn.Sequential(
            nn.Linear(self.num_inputs,
                      self.num_hiddens),
            nn.ReLU(),
            nn.Linear(self.num_hiddens,
                      self.num_outputs)
        )
        self.critic_q2 = nn.Sequential(
            nn.Linear(self.num_inputs,
                      self.num_hiddens),
            nn.ReLU(),
            nn.Linear(self.num_hiddens,
                      self.num_outputs)
        )

    def forward(self, hidden_state_tp1, action):
        """Forward pass."""
        critic_input = torch.cat([hidden_state_tp1, action], dim=-1)

        # Estimate the reward based on the hidden state and the action taken
        q1_predicted_reward = self.critic_q1(critic_input)
        q2_predicted_reward = self.critic_q2(critic_input)

        return q1_predicted_reward, q2_predicted_reward
