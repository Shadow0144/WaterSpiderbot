"""Locomotion policy based on a Soft Actor-Critics Deep Neural Network."""

import torch
from torch import nn

from .deep_actor_critic_policy import DeepActorCriticPolicy
from .deep_critic import DeepCritic


class DeepSoftActorCriticsPolicy(DeepActorCriticPolicy):
    """Provides an action based on the current state."""

    def __init__(self, logger):
        """Initialize the locomotion neural network."""
        super().__init__(logger)

        # Entropy temperature scaling
        self.alpha = 0.2

    def _create_critics(self):
        """Create the Critic(s) and their optimizer(s)."""
        self.num_critic_inputs = (
            self.num_actor_recurrent_hiddens + self.num_actor_outputs
        )
        self.critics = [
            DeepCritic(
                self.num_critic_inputs,
                self.num_critic_hiddens,
                self.num_critic_outputs
            ).to(self.device),
            DeepCritic(
                self.num_critic_inputs,
                self.num_critic_hiddens,
                self.num_critic_outputs
            ).to(self.device)
        ]
        self.critic_optimizers = [
            torch.optim.Adam(
                self.critics[0].parameters(),
                lr=1e-4),
            torch.optim.Adam(
                self.critics[1].parameters(),
                lr=1e-4)
        ]

    def _train_actor_critic_step(self,
                                 transition_t,
                                 state_tp1,
                                 reward_t,
                                 training_done):
        """Perform a single-step Soft Actor-Critic update."""
        self.actor.train()
        for critic in self.critics:
            critic.train()

        reward_tensor = torch.tensor(
                    [reward_t],
                    dtype=torch.float32,
                    device=self.device
        ).view(-1)

        # Compute Bellman Target (t+1)

        with torch.no_grad():
            if training_done:
                target_value = reward_tensor
            else:
                action_distribution_tp1, hidden_state_tp2 = (
                    self.actor(
                        state_tp1,
                        transition_t.hidden_state_tp1
                    )
                )
                action_tp1 = action_distribution_tp1.sample()
                log_probability_tp1 = action_distribution_tp1.log_prob(
                    action_tp1
                ).sum(dim=-1)

                critic_value_tp1_min = None
                for critic in self.critics:
                    critic_value_tp1 = (
                        critic(hidden_state_tp2, action_tp1)
                    ).view(-1)
                    if critic_value_tp1_min is None:
                        critic_value_tp1_min = critic_value_tp1
                    else:
                        critic_value_tp1_min = (
                            torch.min(critic_value_tp1_min, critic_value_tp1)
                        )

                critic_value_tp1_min -= self.alpha * log_probability_tp1
                target_value = (
                    reward_tensor + self.gamma * critic_value_tp1_min
                )

        # Update Critics

        for critic, critic_optimizer in zip(
            self.critics, self.critic_optimizers
        ):
            critic_value_t = critic(
                transition_t.hidden_state_t, transition_t.action_t
            ).view(-1)
            critic_loss = self.loss_function(critic_value_t, target_value)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_optimizer.step()

        # Update Actor

        pi_action_distribution_t, _ = (
            self.actor(transition_t.state_t, transition_t.hidden_state_t)
        )
        pi_action_t = pi_action_distribution_t.rsample()
        pi_log_probability_t = (
            pi_action_distribution_t.log_prob(
                pi_action_t
            ).sum(dim=-1)
        )

        critic_value_t_min = None
        for critic in self.critics:
            pi_critic_value_t = (
                critic(transition_t.hidden_state_t, pi_action_t)
            ).view(-1)
            if critic_value_t_min is None:
                critic_value_t_min = pi_critic_value_t
            else:
                critic_value_t_min = (
                    torch.min(critic_value_t_min, pi_critic_value_t)
                )

        actor_loss = (
            self.alpha * pi_log_probability_t - critic_value_t_min
        ).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
