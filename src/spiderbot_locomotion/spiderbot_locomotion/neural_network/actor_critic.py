"""Locomotion actor-critic."""

import torch
from torch import nn
from torch.distributions import Normal


class LocomotionActorCritic(nn.Module):
    """Locomotion actor-critic."""

    def __init__(self):
        """Initialize the locomotion actor-critic."""
        super().__init__()

        # Inputs:
        #  target_x, target_y, target_theta,
        #  body_x, body_y, body_z,
        #  body_rx, body_ry, body_rz, body_rw,
        #  body_vx, body_vy, body_vz
        #  body_vrp, body_vrr, body_vry,
        #  foreach leg: # 8
        #   leg_q1, leg_q2, leg_q3,
        #   leg_vq1, leg_vq2, leg_vq3,
        #   leg_x, leg_y, leg_z,
        #   leg_rx, leg_ry, leg_rz
        # Outputs:
        # foreach leg: # 8
        #  leg_q1, leg_q2, leg_q3
        self.num_inputs = 3 + 13 + (8 * 12)  # 112
        self.num_outputs = (8 * 3)  # 24
        self.hidden_dim = 256

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.num_inputs, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid()
        )

        # Recurrent layer
        self.gru_cell = nn.GRUCell(512, self.hidden_dim)

        self.actor_mean = nn.Linear(self.hidden_dim, self.num_outputs)
        self.actor_log_std = nn.Parameter(torch.zeros(self.num_outputs))
        self.critic_value = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, hidden_state_t=None):
        """Forward pass."""
        # If no hidden state yet, initialize with zeros
        if hidden_state_t is None:
            hidden_state_t = torch.zeros(
                x.size(0),
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype
            )

        # Get the features of the current input (x)
        features = self.feature_extractor(x)
        # Update the estimate with information from the previous state
        hidden_state_tp1 = self.gru_cell(features, hidden_state_t)

        # Get a distribution of possible actions (actuator angles) to take
        mean = self.actor_mean(hidden_state_tp1)
        log_std = torch.clamp(self.actor_log_std, min=-20, max=2)
        std = torch.exp(log_std)
        action_dist = Normal(mean, std)

        # Estimate the reward based on the predicted next state
        value = self.critic_value(hidden_state_tp1)

        return action_dist, value, hidden_state_tp1
