"""Locomotion Actor using a Deep Neural Network."""

import torch
from torch import nn
from torch.distributions import Normal


class DeepActor(nn.Module):
    """Locomotion Actor using a Deep Neural Network."""

    def __init__(
            self,
            num_inputs,
            num_feature_hiddens,
            num_outputs):
        """Initialize the Actor."""
        super().__init__()

        self.num_inputs = num_inputs
        self.num_feature_hiddens = num_feature_hiddens
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

        self.actor_mean = nn.Linear(self.num_feature_hiddens,
                                    self.num_outputs)
        self.actor_log_std = nn.Parameter(torch.zeros(self.num_outputs))

    def forward(self, state_t):
        """Forward pass."""
        # Get the features of the current state
        features_t = self.feature_extractor(state_t)

        # Get a distribution of possible actions (actuator angles) to take
        mean_t = self.actor_mean(features_t)
        log_std_t = torch.clamp(self.actor_log_std, min=-20, max=2)
        std_t = torch.exp(log_std_t)
        action_distribution_t = Normal(mean_t, std_t)

        return action_distribution_t
