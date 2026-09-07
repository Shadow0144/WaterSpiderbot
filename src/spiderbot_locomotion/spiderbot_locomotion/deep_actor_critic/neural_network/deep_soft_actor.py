"""Locomotion Soft Actor using a Deep Neural Network."""

import torch
from torch import nn
from torch.distributions import Normal


class DeepSoftActor(nn.Module):
    """Locomotion Soft Actor using a Deep Neural Network."""

    def __init__(
            self,
            num_inputs,
            num_feature_hiddens,
            num_recurrent_hiddens,
            num_outputs):
        """Initialize the Soft Actor."""
        super().__init__()

        self.num_inputs = num_inputs
        self.num_feature_hiddens = num_feature_hiddens
        self.num_recurrent_hiddens = num_recurrent_hiddens
        self.num_outputs = num_outputs

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.num_inputs,
                      self.num_feature_hiddens),
            nn.Sigmoid(),
            nn.Linear(self.num_feature_hiddens,
                      self.num_feature_hiddens),
            nn.ReLU()
        )

        # Recurrent layer
        self.gru_cell = nn.GRUCell(self.num_feature_hiddens,
                                   self.num_recurrent_hiddens)

        self.actor_mean = nn.Linear(self.num_recurrent_hiddens,
                                    self.num_outputs)
        self.actor_log_std = nn.Parameter(torch.zeros(self.num_outputs))

    def forward(self, state_t, hidden_state_t=None):
        """Forward pass."""
        # If no hidden state yet, initialize with zeros
        if hidden_state_t is None:
            hidden_state_t = torch.zeros(
                state_t.size(0),
                self.num_recurrent_hiddens,
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

        return action_dist, hidden_state_tp1
