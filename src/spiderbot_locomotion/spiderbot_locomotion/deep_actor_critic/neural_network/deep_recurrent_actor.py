"""Locomotion Actor using a Deep Recurrent Neural Network."""

import torch
from torch import nn
from torch.distributions import Normal


class DeepRecurrentActor(nn.Module):
    """Locomotion Actor using a Deep Recurrent Neural Network."""

    def __init__(
            self,
            num_inputs,
            num_feature_hiddens,
            num_recurrent_hiddens,
            num_outputs):
        """Initialize the Actor."""
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
        # If there is no hidden state yet, initialize the layer with zeros
        if hidden_state_t is None:
            hidden_state_t = torch.zeros(
                state_t.size(0),
                self.num_recurrent_hiddens,
                device=state_t.device,
                dtype=state_t.dtype
            )

        # Get the features of the current state
        features_t = self.feature_extractor(state_t)
        # Update the estimate with information from the previous state
        hidden_state_tp1 = self.gru_cell(features_t, hidden_state_t)

        # Get a distribution of possible actions (actuator angles) to take
        mean_t = self.actor_mean(hidden_state_tp1)
        log_std_t = torch.clamp(self.actor_log_std, min=-20, max=2)
        std_t = torch.exp(log_std_t)
        action_distribution_t = Normal(mean_t, std_t)

        return action_distribution_t, hidden_state_tp1
