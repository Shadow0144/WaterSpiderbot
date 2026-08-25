"""Locomotion Actor-Critic using a Deep Neural Network."""

import torch
from torch import nn
from torch.distributions import Normal


class DeepActorCritic(nn.Module):
    """Locomotion Actor-Critic using a Deep Neural Network."""

    def __init__(self):
        """Initialize the Actor-Critic."""
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
        self.feature_hidden_dim = 512
        self.recurrent_hidden_dim = 256
        self.critic_num_outputs = 1

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.num_inputs,
                      self.feature_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.feature_hidden_dim,
                      self.feature_hidden_dim),
            nn.ReLU()
        )

        # Recurrent layer
        self.gru_cell = nn.GRUCell(self.feature_hidden_dim,
                                   self.recurrent_hidden_dim)

        self.actor_mean = nn.Linear(self.recurrent_hidden_dim,
                                    self.num_outputs)
        self.actor_log_std = nn.Parameter(torch.zeros(self.num_outputs))

        self.critic_value = nn.Linear(self.recurrent_hidden_dim,
                                      self.critic_num_outputs)

    def forward(self, state_t, hidden_state_t=None):
        """Forward pass."""
        # If no hidden state yet, initialize with zeros
        if hidden_state_t is None:
            hidden_state_t = torch.zeros(
                state_t.size(0),
                self.recurrent_hidden_dim,
                device=state_t.device,
                dtype=state_t.dtype
            )

        # Get the features of the current state
        features = self.feature_extractor(state_t)
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
