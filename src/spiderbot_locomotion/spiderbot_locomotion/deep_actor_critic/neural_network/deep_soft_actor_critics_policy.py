"""Locomotion policy based on a Soft Actor-Critics Deep Neural Network."""

import torch
from torch import nn

from .checkpoint_file_manager import CheckpointFileManager
from .deep_actor import DeepActor
from .deep_critic import DeepCritic
from .reward_calculator import RewardCalculator
from .step_transition import StepTransition
from .utility import construct_input_vector


class DeepSoftActorCriticsPolicy():
    """Provides a distribution of actions for the current state."""

    def __init__(self, logger):
        """Initialize the locomotion neural network."""
        self.logger = logger

        self.device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available() else
            'cpu'
        )

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
        self.num_actor_inputs = 3 + 13 + (8 * 12)  # 112
        self.num_actor_feature_hiddens = 512
        self.num_actor_recurrent_hiddens = 256
        self.num_actor_outputs = (8 * 3)  # 24

        self.num_critic_inputs = (
            self.num_actor_recurrent_hiddens + self.num_actor_outputs
        )
        self.num_critic_hiddens = 256
        self.num_critic_outputs = 1

        self.actor = DeepActor(
            self.num_actor_inputs,
            self.num_actor_feature_hiddens,
            self.num_actor_recurrent_hiddens,
            self.num_actor_outputs
        ).to(self.device)
        self.critic1 = DeepCritic(
            self.num_critic_inputs,
            self.num_critic_hiddens,
            self.num_critic_outputs
        ).to(self.device)
        self.critic2 = DeepCritic(
            self.num_critic_inputs,
            self.num_critic_hiddens,
            self.num_critic_outputs
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=1e-4)
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(),
            lr=1e-4)
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(),
            lr=1e-4)
        self.loss_fn = nn.MSELoss()

        self.checkpoint_file_manager = CheckpointFileManager()

        # Recurrent hidden state (initialize with zeros)
        self.hidden_state_t = torch.zeros(
            1,
            self.num_actor_recurrent_hiddens,
            device=self.device
        )

        self.reward_calculator = RewardCalculator()

        # Entropy temperature scaling
        self.alpha = 0.2

        # Reward horizon scaling
        self.gamma = 0.99

        # Target information
        self.target = None

        # Previous state information
        self.transition_t = None

    def start_new_training_episode(self):
        """Reset the internal state variables for the episode."""
        self.hidden_state_t = torch.zeros(
            1,
            self.num_actor_recurrent_hiddens,
            device=self.device
        )
        self.transition_t = None
        self.reward_calculator.start_new_training_episode()

    def get_model_weights_exists(self, filename='test_weights.pt'):
        """Get if the model weight file exists."""
        return self.checkpoint_file_manager.get_model_weights_exists(
            filename
        )

    def save_weights(self, filename='test_weights.pt'):
        """Save the learned weights to a file."""
        self.logger.info(f'Saving weights: {filename}')
        try:
            self.checkpoint_file_manager.save_soft_actor_critics_weights(
                filename,
                self.actor,
                self.critic1,
                self.critic2,
                self.actor_optimizer,
                self.critic1_optimizer,
                self.critic2_optimizer
            )
            self.logger.info(f'Saved weights: {filename}')
        except RuntimeError:
            self.logger.warning('Failed to save weights')

    def load_weights(self, filename='test_weights.pt'):
        """Load the learned weights from a file."""
        try:
            self.logger.info(f'Loading weights: {filename}')
            if self.get_model_weights_exists(filename):
                self.checkpoint_file_manager.load_soft_actor_critics_weights(
                    filename,
                    self.actor,
                    self.critic1,
                    self.critic2,
                    self.actor_optimizer,
                    self.critic1_optimizer,
                    self.critic2_optimizer,
                    self.device
                )
                self.logger.info(f'Loaded weights: {filename}')
                self.start_new_training_episode()
        except RuntimeError:
            self.logger.warning('Failed to load weights')

    def delete_saved_weights(self, filename):
        """Delete the saved weights file."""
        if self.checkpoint_file_manager.get_model_weights_exists(filename):
            self.checkpoint_file_manager.delete_saved_weights(filename)
            self.logger.info(f'Deleted weights: {filename}')

    def reset_learned_weights(self):
        """Backup the current weights and start with new random weights."""
        self.checkpoint_file_manager.reset_learned_soft_actor_critics_weights()
        self.actor = DeepActor(
            self.num_actor_inputs,
            self.num_actor_feature_hiddens,
            self.num_actor_recurrent_hiddens,
            self.num_actor_outputs
            ).to(self.device)
        self.critic1 = DeepCritic(
            self.num_critic_inputs,
            self.num_critic_hiddens,
            self.num_critic_outputs
            ).to(self.device)
        self.critic2 = DeepCritic(
            self.num_critic_inputs,
            self.num_critic_hiddens,
            self.num_critic_outputs
            ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=1e-4)
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(),
            lr=1e-4)
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(),
            lr=1e-4)
        self.start_new_training_episode()

    def set_target(self, time_to_reach_target_s, target):
        """Update the target and the time expected to reach the target."""
        self.reward_calculator.set_time_to_reach_target(time_to_reach_target_s)
        self.target = target

    def get_episode_reward(self):
        """Return the average reward rate per second for the episode."""
        return (
            0.0
            if self.reward_calculator.time_to_reach_target_s == 0.0 else
            self.reward_calculator.episode_reward /
            self.reward_calculator.time_to_reach_target_s
        )

    def select_action(self, spiderbot_pose, deterministic=False):
        """Select the next action."""
        if self.target is None:
            return None  # Exit early if there is no target

        state_t = construct_input_vector(
                self.target,
                spiderbot_pose,
                self.device
        )

        if deterministic:
            action_t = self._select_action_deterministic(state_t)
        else:
            action_t = self._select_action_stochastic(state_t)
        return action_t

    def _select_action_deterministic(self, state_t):
        """Step execution for deployment."""
        self.actor.eval()
        with torch.no_grad():
            action_distribution_t, hidden_state_tp1 = self.actor(
                state_t, self.hidden_state_t
            )
            self.hidden_state_t = hidden_state_tp1
            action_t_np = action_distribution_t.mean.squeeze(0).cpu().numpy()
            return action_t_np

    def _select_action_stochastic(self, state_t):
        """Step execution for training."""
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

        action_distribution_t, hidden_state_tp1 = self.actor(
            state_t, self.hidden_state_t
        )

        action_t = action_distribution_t.sample()
        action_t_np = action_t.squeeze(0).detach().cpu().numpy()

        log_probability_t = action_distribution_t.log_prob(
            action_t
        ).sum(dim=-1)

        hidden_state_t = self.hidden_state_t
        hidden_state_tp1 = hidden_state_tp1.detach()
        self.hidden_state_t = hidden_state_tp1

        transition_t = StepTransition(
            state_t,
            action_t,
            log_probability_t,
            hidden_state_t,
            hidden_state_tp1
        )

        return (
            action_t_np,
            transition_t
        )

    def train_step(self,
                   spiderbot_pose,
                   delta_time):
        """Perform a single step of training."""
        reward_t = 0.0
        training_done = False

        if self.target is None:
            return None, reward_t, training_done  # Return early

        state_t = construct_input_vector(
                self.target,
                spiderbot_pose,
                self.device
        )

        if self.transition_t is not None:
            # If there was a previous state,
            # calculate the reward and train the actor-critic
            reward_t, training_done = (
                self.reward_calculator.compute_step_reward(
                    self.target,
                    spiderbot_pose,
                    delta_time
                )
            )

            self._train_actor_critic_step(
                self.transition_t,
                state_t,
                reward_t,
                training_done
            )

        action_t, self.transition_t = (
            self._select_action_stochastic(
                state_t
            )
        )

        return action_t, reward_t, training_done

    def _train_actor_critic_step(self,
                                 transition_t,
                                 state_tp1,
                                 reward_t,
                                 training_done):
        """Perform a single-step Soft Actor-Critic update."""
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

        reward_tensor = torch.tensor([reward_t],
                                     dtype=torch.float32,
                                     device=self.device)

        # Compute Bellman Target (t+1)

        with torch.no_grad():
            if training_done:
                target_value = reward_tensor
            else:
                action_distribution_tp1, _ = (
                    self.actor(
                        state_tp1,
                        transition_t.hidden_state_tp1
                    )
                )
                action_tp1 = action_distribution_tp1.sample()
                log_probability_tp1 = action_distribution_tp1.log_prob(
                    action_tp1
                ).sum(dim=-1)
                critic1_value_tp1 = (
                    self.critic1(transition_t.hidden_state_tp1, action_tp1)
                ).view(-1)
                critic2_value_tp1 = (
                    self.critic2(transition_t.hidden_state_tp1, action_tp1)
                ).view(-1)
                critic_value_tp1_min = (
                    torch.min(critic1_value_tp1, critic2_value_tp1) - (
                        self.alpha * log_probability_tp1
                    )
                )
                target_value = (
                    reward_tensor + self.gamma * critic_value_tp1_min
                )

        # Update Critics

        critic1_value_t = self.critic1(
            transition_t.hidden_state_t, transition_t.action_t
        ).view(-1)
        critic2_value_t = self.critic2(
            transition_t.hidden_state_t, transition_t.action_t
        ).view(-1)

        critic1_loss = self.loss_fn(critic1_value_t, target_value)
        critic2_loss = self.loss_fn(critic2_value_t, target_value)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic2_optimizer.step()

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

        pi_critic1_value_t = self.critic1(
            transition_t.hidden_state_t, pi_action_t
        ).view(-1)
        pi_critic2_value_t = self.critic2(
            transition_t.hidden_state_t, pi_action_t
        ).view(-1)
        critic_value_t_min = torch.min(pi_critic1_value_t, pi_critic2_value_t)

        actor_loss = (
            self.alpha * pi_log_probability_t - critic_value_t_min
        ).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return actor_loss.item() + critic1_loss.item() + critic2_loss.item()
