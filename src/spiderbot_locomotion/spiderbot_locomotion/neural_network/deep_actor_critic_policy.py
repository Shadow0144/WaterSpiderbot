"""Locomotion policy based on an Actor-Critic Deep Neural Network."""

import torch
from torch import nn

from .checkpoint_file_manager import CheckpointFileManager
from .deep_actor_critic import DeepActorCritic
from .reward_calculator import RewardCalculator
from .utility import construct_input_vector


class DeepActorCriticPolicy():
    """Provides a distribution of actions for the current state."""

    class IterationState:
        """Store training state information from a step."""

        def __init__(self,
                     action_np=None,
                     state_t=None,
                     log_probability_t=None,
                     value_t=None,
                     hidden_state_t=None,
                     hidden_state_tp1=None
                     ):
            """Store the values."""
            self.action_np = action_np
            self.state_t = state_t
            self.log_probability_t = log_probability_t
            self.value_t = value_t
            self.hidden_state_t = hidden_state_t
            self.hidden_state_tp1 = hidden_state_tp1

    def __init__(self):
        """Initialize the locomotion neural network."""
        self.device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available() else
            'cpu'
        )

        self.actor_critic = DeepActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=1e-4)
        self.loss_fn = nn.MSELoss()

        self.checkpoint_file_manager = CheckpointFileManager()

        # Recurrent hidden state
        self.hidden_state = None

        self.reward_calculator = RewardCalculator()

        # Reward horizon scaling
        self.gamma = 0.99

        # Target information
        self.target = None

        # Previous state information
        self.latest_iter_state = None

    def reset(self):
        """Reset the internal state variables."""
        self.hidden_state = None
        self.latest_iter_state = None
        self.reward_calculator.reset()

    def get_model_weights_exist(self, filename='test_weights.pt'):
        """Get if the model weight file exists."""
        return self.checkpoint_file_manager.get_model_weights_exist(
            filename
        )

    def save_weights(self, filename='test_weights.pt'):
        """Save the learned weights to a file."""
        self.checkpoint_file_manager.save_weights(
            self.actor_critic,
            self.optimizer,
            filename
        )

    def load_weights(self, filename='test_weights.pt'):
        """Load the learned weights from a file."""
        self.checkpoint_file_manager.load_weights(
            self.actor_critic,
            self.optimizer,
            self.device,
            filename
        )

    def reset_learned_weights(self):
        """Backup the current weights and start with new random weights."""
        (
            self.actor_critic,
            self.optimizer
        ) = self.checkpoint_file_manager.reset_learned_weights()

    def delete_saved_weights(self, filename='test_weights.pt'):
        """Delete the saved weights file."""
        self.checkpoint_file_manager.delete_saved_weights(filename)

    def set_target(self, time_to_goal_s, target):
        """Update the target and the time expected to reach the goal."""
        self.reward_calculator.set_time_to_goal(time_to_goal_s)
        self.target = target

    def get_episode_reward(self):
        """Return the total reward of the episode."""
        return (
            0.0
            if self.reward_calculator.time_to_goal_s == 0.0 else
            self.reward_calculator.episode_reward /
            self.reward_calculator.time_to_goal_s
        )

    def select_action(self, spiderbot_pose, deterministic=False):
        """Select the next action."""
        if deterministic:
            next_action = self.select_action_deterministic(spiderbot_pose)
        else:
            state_iteration = self.select_action_stochastic(spiderbot_pose)
            if state_iteration is not None:
                next_action = state_iteration.action_np
            else:
                next_action = None
        return next_action

    def select_action_deterministic(self, spiderbot_pose):
        """Step execution for deployment."""
        if self.target is None:
            return None  # Exit early if there is no target

        self.actor_critic.eval()
        with torch.no_grad():
            state_tensor = construct_input_vector(
                self.target,
                spiderbot_pose,
                self.device
            )
            action_dist, _, hidden_state_tp1 = self.actor_critic(
                state_tensor, self.hidden_state
            )
            self.hidden_state = hidden_state_tp1
            return action_dist.mean.squeeze(0).cpu().numpy()

    def select_action_stochastic(self, spiderbot_pose):
        """Step execution for training."""
        if self.target is None:
            return None  # Exit early if there is no target

        self.actor_critic.train()
        state_tensor = construct_input_vector(
            self.target,
            spiderbot_pose,
            self.device
        )

        action_dist, value_t, hidden_state_tp1 = self.actor_critic(
            state_tensor,
            self.hidden_state
        )

        action = action_dist.sample()
        log_probability = action_dist.log_prob(action).sum(dim=-1)

        action_np = action.squeeze(0).detach().cpu().numpy()

        hidden_state_t = self.hidden_state
        self.hidden_state = hidden_state_tp1.detach()

        return self.IterationState(
            action_np,
            state_tensor,
            log_probability,
            value_t,
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

            self.train_actor_critic_step(
                self.latest_iter_state,
                next_data,
                reward,
                done
            )

        self.latest_iter_state = (
            self.select_action_stochastic(
                spiderbot_pose
            )
        )

        return self.latest_iter_state.action_np, reward, done

    def train_actor_critic_step(self,
                                iter_state,
                                state_tp1_tensor,
                                reward,
                                done):
        """Perform a single-step Actor-Critic update."""
        self.actor_critic.train()

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
            _, value_tp1, _ = self.actor_critic(state_tp1_tensor,
                                                hidden_state_for_next)

        target_value = (
            reward_tensor + (self.gamma * value_tp1.squeeze(0) * done_mask)
        )

        advantage = target_value - iter_state.value_t.squeeze(0)

        actor_loss = -iter_state.log_probability_t * advantage.detach()

        critic_loss = self.loss_fn(iter_state.value_t.squeeze(0), target_value)

        total_loss = actor_loss + (0.5 * critic_loss)

        self.optimizer.zero_grad()
        total_loss.backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item()
