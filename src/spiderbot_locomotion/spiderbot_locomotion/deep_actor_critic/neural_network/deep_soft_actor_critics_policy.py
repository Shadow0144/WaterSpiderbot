"""Locomotion policy based on an Soft Actor-Critics Deep Neural Network."""

import torch
from torch import nn

from .checkpoint_file_manager import CheckpointFileManager
from .deep_soft_actor import DeepSoftActor
from .deep_soft_critics import DeepSoftCritics
from .reward_calculator import RewardCalculator
from .utility import construct_input_vector


class DeepSoftActorCriticsPolicy():
    """Provides a distribution of actions for the current state."""

    class IterationState:
        """Store training state information from a step."""

        def __init__(self,
                     action_tensor=None,
                     action_np=None,
                     state_t=None,
                     log_probability_t=None,
                     hidden_state_t=None,
                     hidden_state_tp1=None
                     ):
            """Store the values."""
            self.action_tensor = action_tensor
            self.action_np = action_np
            self.state_t = state_t
            self.log_probability_t = log_probability_t
            self.hidden_state_t = hidden_state_t
            self.hidden_state_tp1 = hidden_state_tp1

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

        self.actor = DeepSoftActor(
            self.num_actor_inputs,
            self.num_actor_feature_hiddens,
            self.num_actor_recurrent_hiddens,
            self.num_actor_outputs
        ).to(self.device)
        self.critics = DeepSoftCritics(
            self.num_critic_inputs,
            self.num_critic_hiddens,
            self.num_critic_outputs
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=1e-4)
        self.critics_optimizer = torch.optim.Adam(
            self.critics.parameters(),
            lr=1e-4)
        self.loss_fn = nn.MSELoss()

        self.checkpoint_file_manager = CheckpointFileManager()

        # Recurrent hidden state
        self.hidden_state = None

        self.reward_calculator = RewardCalculator()

        # Entropy temperature scaling
        self.alpha = 0.2

        # Reward horizon scaling
        self.gamma = 0.99

        # Target information
        self.target = None

        # Previous state information
        self.latest_iter_state = None

    def start_new_training_episode(self):
        """Reset the internal state variables for the episode."""
        self.hidden_state = None
        self.latest_iter_state = None
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
                self.critics,
                self.actor_optimizer,
                self.critics_optimizer
            )
            self.logger.info(f'Saved weights: {filename}')
        except RuntimeError:
            self.logger.warn('Failed to save weights')

    def load_weights(self, filename='test_weights.pt'):
        """Load the learned weights from a file."""
        try:
            self.logger.info(f'Loading weights: {filename}')
            if self.get_model_weights_exists(filename):
                self.checkpoint_file_manager.load_soft_actor_critics_weights(
                    filename,
                    self.actor,
                    self.critics,
                    self.actor_optimizer,
                    self.critics_optimizer,
                    self.device
                )
                self.logger.info(f'Loaded weights: {filename}')
                self.start_new_training_episode()
        except RuntimeError:
            self.logger.warn('Failed to load weights')

    def delete_saved_weights(self, filename):
        """Delete the saved weights file."""
        if self.checkpoint_file_manager.get_model_weights_exists(filename):
            self.checkpoint_file_manager.delete_saved_weights(filename)
            self.logger.info(f'Deleted weights: {filename}')

    def reset_learned_weights(self):
        """Backup the current weights and start with new random weights."""
        self.checkpoint_file_manager.reset_learned_soft_actor_critics_weights()
        self.actor = DeepSoftActor(
            self.num_actor_inputs,
            self.num_actor_feature_hiddens,
            self.num_actor_recurrent_hiddens,
            self.num_actor_outputs
            ).to(self.device)
        self.critics = DeepSoftCritics(
            self.num_critic_inputs,
            self.num_critic_hiddens,
            self.num_critic_outputs
            ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=1e-4)
        self.critics_optimizer = torch.optim.Adam(
            self.critics.parameters(),
            lr=1e-4)
        self.start_new_training_episode()

    def set_target(self, time_to_reach_target_s, target):
        """Update the target and the time expected to reach the target."""
        self.reward_calculator.set_time_to_reach_target(time_to_reach_target_s)
        self.target = target

    def get_episode_reward(self):
        """Return the total reward of the episode."""
        return (
            0.0
            if self.reward_calculator.time_to_reach_target_s == 0.0 else
            self.reward_calculator.episode_reward /
            self.reward_calculator.time_to_reach_target_s
        )

    def select_action(self, spiderbot_pose, deterministic=False):
        """Select the next action."""
        if deterministic:
            next_action = self._select_action_deterministic(spiderbot_pose)
        else:
            state_iteration = self._select_action_stochastic(spiderbot_pose)
            if state_iteration is not None:
                next_action = state_iteration.action_np
            else:
                next_action = None
        return next_action

    def _select_action_deterministic(self, spiderbot_pose):
        """Step execution for deployment."""
        if self.target is None:
            return None  # Exit early if there is no target

        self.actor.eval()
        with torch.no_grad():
            state_tensor = construct_input_vector(
                self.target,
                spiderbot_pose,
                self.device
            )
            action_dist, hidden_state_tp1 = self.actor(
                state_tensor, self.hidden_state
            )
            self.hidden_state = hidden_state_tp1
            return action_dist.mean.squeeze(0).cpu().numpy()

    def _select_action_stochastic(self, spiderbot_pose):
        """Step execution for training."""
        if self.target is None:
            return None  # Exit early if there is no target

        self.actor.train()
        self.critics.train()
        state_tensor = construct_input_vector(
            self.target,
            spiderbot_pose,
            self.device
        )

        action_dist, hidden_state_tp1 = self.actor(
            state_tensor,
            self.hidden_state
        )

        action_tensor = action_dist.sample()
        log_probability = action_dist.log_prob(action_tensor).sum(dim=-1)

        action_np = action_tensor.squeeze(0).detach().cpu().numpy()

        action_tensor_t = action_tensor.detach()
        state_tensor_t = state_tensor.detach()

        hidden_state_t = self.hidden_state
        hidden_state_tp1 = hidden_state_tp1.detach()
        self.hidden_state = hidden_state_tp1

        return self.IterationState(
            action_tensor_t,
            action_np,
            state_tensor_t,
            log_probability,
            hidden_state_t,
            hidden_state_tp1
        )

    def train_step(self,
                   spiderbot_pose,
                   delta_time):
        """Perform a single step of training."""
        reward = 0.0
        done = False

        if self.target is None:
            return None, reward, done  # Return early

        if self.latest_iter_state is not None:
            # If there was a previous iter_state,
            # calculate the reward and train the actor-critic
            reward, done = (
                self.reward_calculator.compute_step_reward(
                    self.target,
                    spiderbot_pose,
                    delta_time
                )
            )

            next_data = construct_input_vector(
                self.target,
                spiderbot_pose,
                self.device
            )

            self._train_actor_critic_step(
                self.latest_iter_state,
                next_data,
                reward,
                done
            )

        self.latest_iter_state = (
            self._select_action_stochastic(
                spiderbot_pose
            )
        )

        return self.latest_iter_state.action_np, reward, done

    def _train_actor_critic_step(self,
                                 iter_state,
                                 state_tp1_tensor,
                                 reward,
                                 done):
        """Perform a single-step Actor-Critic update."""
        self.actor.train()
        self.critics.train()

        reward_tensor = torch.tensor([reward],
                                     dtype=torch.float32,
                                     device=self.device)
        done_mask = torch.tensor([0.0 if done else 1.0],
                                 dtype=torch.float32,
                                 device=self.device)

        with torch.no_grad():
            hidden_state_for_next = (
                None if done else iter_state.hidden_state_tp1
            )
            next_action_dist, _ = self.actor(state_tp1_tensor,
                                             hidden_state_for_next)
            next_action = next_action_dist.sample()
            next_log_prob = next_action_dist.log_prob(next_action).sum(
                dim=-1, keepdim=True
            )
            target_q1, target_q2 = (
                self.critics(iter_state.hidden_state_tp1, next_action)
            )
            target_q_min = (
                torch.min(target_q1, target_q2) - (self.alpha * next_log_prob)
            )
            y = reward_tensor + (done_mask * self.gamma * target_q_min)

        q1_prediction, q2_prediction = self.critics(
            iter_state.hidden_state_tp1, iter_state.action_tensor
        )

        critics_loss = (
            self.loss_fn(q1_prediction, y) + self.loss_fn(q2_prediction, y)
        )

        self.critics_optimizer.zero_grad()
        critics_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), max_norm=1.0)
        self.critics_optimizer.step()

        pi_action_dist, _ = (
            self.actor(iter_state.state_t, iter_state.hidden_state_t)
        )
        pi_action = pi_action_dist.rsample()
        pi_log_prob = (
            pi_action_dist.log_prob(pi_action).sum(dim=-1, keepdim=True)
        )

        q1_pi, q2_pi = self.critics(iter_state.hidden_state_tp1, pi_action)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * pi_log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return actor_loss.item() + critics_loss.item()
