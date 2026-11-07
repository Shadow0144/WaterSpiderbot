"""Locomotion policy based on a Soft Actor-Critics Deep Neural Network."""

import copy
import random
from collections import deque

import torch
from torch import nn

from .training_observation import TrainingObservation
from ..file_managers.model_weights_file_manager import ModelWeightsFileManager
from ..neural_network.deep_actor import DeepActor
from ..neural_network.deep_critic import DeepCritic
from ..reward_functions.canonical_pose_reward_function import (
    CanonicalPoseRewardFunction
)
from ..reward_functions.complex_reward_function import ComplexRewardFunction
from ..reward_functions.training_status import TrainingStatus
from ..utility import construct_input_vector


class DeepSoftActorCriticsPolicy():
    """Provides an action based on the current state."""

    def __init__(
            self,
            logger,
            model_weights_filename
            ):
        """Initialize the locomotion neural network."""
        self.logger = logger
        self.model_weights_filename = model_weights_filename

        self.device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available() else
            'cpu'
        )

        self._determine_layer_sizes()
        self._create_actor()
        self._create_critics()
        self.loss_function = nn.MSELoss()

        self.model_weights_file_manager = ModelWeightsFileManager(
            model_weights_filename
        )

        use_canonical_pose = True
        if not use_canonical_pose:
            self.reward_function = ComplexRewardFunction(self.logger)
        else:
            self.reward_function = CanonicalPoseRewardFunction(self.logger)

        # Reward horizon scaling
        self.gamma = 0.99

        # Target information
        self.target = None

        self.are_actions_normalized = True

        self.batch_size = 100
        self.buffer_size = 10_000
        self.training_observations = deque(maxlen=self.buffer_size)
        self.current_training_observation = None

        # The frames queue is a buffer (of size k) of past observations
        # to concatenate into a state
        self.frames = deque(maxlen=self.num_frames_k)
        self._reset_frame_queue()

        # Entropy temperature scaling
        self.alpha = 0.2

        # Noise term to prevent log of zero in the Jacobian correction
        self.epsilon = 1e-6

        # Polyak soft update factor
        self.tau = 0.005

    def _determine_layer_sizes(self):
        """Set all the layer sizes."""
        # Inputs:
        #  target_x, target_y, target_theta,
        #  body_x, body_y, body_z,
        #  body_rx, body_ry, body_rz, body_rw,
        #  body_vx, body_vy, body_vz
        #  body_vrp, body_vrr, body_vry,
        #  tendon_tension,
        #  for each leg: # 8
        #   leg_q1, leg_q2, leg_q3, leg_q3,
        #   leg_vq1, leg_vq2, leg_vq3, leg_vq3,
        #   claw_x, claw_y, claw_z,
        #   claw_rx, claw_ry, claw_rz,
        #  torque
        # Outputs:
        # tendon_tension,
        # for each leg: # 8
        #  leg_q1, leg_q2, leg_q3, leg_q4

        self.num_frames_k = 10  # Number of frames in each state

        self.num_actor_inputs = 3 + 13 + 1 + (8 * 14) + 1  # 130
        self.num_actor_inputs_k = self.num_actor_inputs * self.num_frames_k
        self.num_actor_feature_hiddens = 512
        self.num_actor_recurrent_hiddens = 256
        self.num_actor_outputs = 1 + (8 * 4)  # 33

        self.num_critic_inputs = (
            self.num_actor_inputs_k + self.num_actor_outputs
        )
        self.num_critic_hiddens = 256
        self.num_critic_outputs = 1

    def _create_actor(self):
        """Create the Actor and its optimizer."""
        self.actor = DeepActor(
            self.num_actor_inputs_k,  # k frames per state
            self.num_actor_feature_hiddens,
            self.num_actor_outputs
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=1e-4)

    def _create_critics(self):
        """Create the Critic(s), Target Critic(s), and their optimizer(s)."""
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

        self.target_critics = copy.deepcopy(self.critics)
        for target_critic in self.target_critics:
            for parameter in target_critic.parameters():
                parameter.requires_grad = False

        self.critic_optimizers = [
            torch.optim.Adam(
                self.critics[0].parameters(),
                lr=1e-4),
            torch.optim.Adam(
                self.critics[1].parameters(),
                lr=1e-4)
        ]

    def reset(self):
        """Reset the state."""
        self.target = None
        self.reward_function.set_target(None)

    def get_are_actions_normalized(self):
        """Return if the poses are normalized or already scaled."""
        return self.are_actions_normalized

    def _reset_frame_queue(self):
        """Zero out all the observed frames in the buffer."""
        self.current_training_observation = None
        self.frames.clear()
        empty_observation = torch.zeros(
            (1, self.num_actor_inputs),
            dtype=torch.float32,
            device=self.device
        )
        for _ in range(self.num_frames_k):
            self.frames.append(empty_observation)

    def start_new_training_episode(self):
        """Reset the internal state variables for the episode."""
        self.current_training_observation = None
        self._reset_frame_queue()
        self.reward_function.start_new_training_episode()

    def _construct_state(self, spiderbot_pose_msg):
        """Construct a state vector from the latest observation."""
        frame_t = construct_input_vector(
                self.target,
                spiderbot_pose_msg,
                self.device
        )
        self.frames.append(frame_t)
        state_t = torch.cat(list(self.frames), dim=-1)
        return state_t

    def save_weights(self, filename_override=None):
        """Save the learned weights to a file."""
        filename = (
            filename_override
            if filename_override else
            self.model_weights_filename
        )
        self.logger.info(f'Saving weights: {filename}')
        try:
            self.model_weights_file_manager.save_model_weights(
                self.actor,
                self.actor_optimizer,
                self.critics,
                self.critic_optimizers,
                filename_override
            )
            self.logger.info(f'Saved weights: {filename}')
        except RuntimeError as e:
            self.logger.warning(f'Failed to save weights: {e}')

    def load_weights(self, filename_override=None):
        """Load the learned weights from a file."""
        try:
            filename = (
                filename_override
                if filename_override else
                self.model_weights_filename
            )
            self.logger.info(f'Loading weights: {filename}')
            if self.model_weights_file_manager.get_model_weights_exists():
                self.model_weights_file_manager.load_model_weights(
                    self.actor,
                    self.actor_optimizer,
                    self.critics,
                    self.critic_optimizers,
                    self.device,
                    filename_override
                )
                self.logger.info(
                    f'Loaded weights: {filename}'
                )
                self.start_new_training_episode()
            else:
                self.logger.warning(
                    f'Weights file {filename} not found'
                )
        except RuntimeError as e:
            self.logger.warning(f'Failed to load weights: {e}')

    def delete_saved_weights(self):
        """Delete the saved weights file."""
        if self.model_weights_file_manager.get_model_weights_exists():
            self.model_weights_file_manager.delete_model_weights()
            self.logger.info(f'Deleted weights: {self.model_weights_filename}')

    def reset_learned_weights(self):
        """Backup the current weights and start with new random weights."""
        self.model_weights_file_manager.reset_model_weights()
        # Create new actors and critics and optimizers with random weights
        self._create_actor()
        self._create_critics()
        self.start_new_training_episode()

    def set_target(self, target):
        """Update the target."""
        self.target = target
        self.reward_function.set_target(self.target)

    def select_action(self, spiderbot_pose_msg, deterministic=False):
        """Select the next action."""
        if self.target is None:
            return None  # Exit early if there is no target

        state_t = self._construct_state(spiderbot_pose_msg)

        if deterministic:
            action_t = self._select_action_deterministic(state_t)
        else:
            action_t, _ = self._select_action_stochastic(state_t)

        return action_t

    def _select_action_deterministic(self, state_t):
        """Step execution for deployment."""
        self.actor.eval()
        with torch.no_grad():
            action_distribution_t = self.actor(state_t)
            action_t = action_distribution_t.mean
            bounded_action_t = torch.tanh(action_t)
            action_t_np = bounded_action_t.squeeze(0).cpu().numpy()
            return action_t_np

    def _select_action_stochastic(self, state_t):
        """Step execution for training."""
        self.actor.train()

        action_distribution_t = self.actor(state_t)

        action_t = action_distribution_t.sample()
        bounded_action_t = torch.tanh(action_t)
        action_t_np = bounded_action_t.squeeze(0).detach().cpu().numpy()

        log_probability_t = action_distribution_t.log_prob(
            action_t
        ).sum(dim=-1)

        return (
            action_t_np,
            action_t,
            log_probability_t
        )

    def train_step(self,
                   spiderbot_pose,
                   delta_time):
        """Perform a single step of training."""
        training_status = TrainingStatus(
            step_reward=0.0,
            target_reached=False,
            episode_terminated=False,
            reward_component_labels=[],
            reward_component_values=[]
        )

        if self.target is None:
            # Return early
            return None, training_status

        state_t = self._construct_state(spiderbot_pose)

        if self.current_training_observation is not None:
            # If there was a previous state,
            # calculate the reward and train the actor-critic
            self.reward_function.compute_step_reward(
                training_status,
                spiderbot_pose,
                delta_time
            )

            # Update the observation with the new state and reward
            # and append it to the list of observations
            self.current_training_observation.update(
                state_t,
                training_status.step_reward,
                training_status.episode_terminated
            )
            self.training_observations.append(
                self.current_training_observation
            )

            if len(self.training_observations) >= self.buffer_size:
                self._train_actor_critic_step()

        action_t_np, action_t, log_probability_t = (
            self._select_action_stochastic(
                state_t
            )
        )

        self.current_training_observation = TrainingObservation(
            state_t,
            action_t,
            log_probability_t
        )

        return action_t_np, training_status

    def _train_actor_critic_step(self):
        """Perform a batch of Soft Actor-Critic updates."""
        self.actor.train()
        for critic in self.critics:
            critic.train()

        batch_observations = random.sample(
            self.training_observations,
            self.batch_size)

        states_t = torch.cat(
            [observation.state_t for observation in batch_observations],
            dim=0).to(self.device).view(self.batch_size, -1)
        actions_t = torch.cat(
            [observation.action_t.view(1, -1)
             for observation in batch_observations],
            dim=0).to(self.device)
        states_tp1 = torch.cat(
            [observation.state_tp1 for observation in batch_observations],
            dim=0).to(self.device).view(self.batch_size, -1)
        rewards_t = torch.tensor(
            [observation.reward_t for observation in batch_observations],
            dtype=torch.float32, device=self.device).unsqueeze(1)
        trainings_done = torch.tensor(
            [observation.episode_terminated
             for observation in batch_observations],
            dtype=torch.float32, device=self.device).unsqueeze(1)

        # Compute Bellman Target (t+1)

        with torch.no_grad():
            action_distributions_tp1 = self.actor(states_tp1)
            actions_tp1 = action_distributions_tp1.sample()
            bounded_pi_actions_tp1 = torch.tanh(actions_tp1)
            log_probability_tp1 = action_distributions_tp1.log_prob(
                bounded_pi_actions_tp1
            ).sum(dim=-1, keepdim=True)

            critic_values_tp1_min = None
            for target_critic in self.target_critics:
                critic_values_tp1 = target_critic(
                    states_tp1, bounded_pi_actions_tp1
                )
                if critic_values_tp1_min is None:
                    critic_values_tp1_min = critic_values_tp1
                else:
                    critic_values_tp1_min = (
                        torch.min(
                            critic_values_tp1_min, critic_values_tp1
                        )
                    )

            critic_values_tp1_min -= self.alpha * log_probability_tp1
            target_values = (
                rewards_t + (
                    (1.0 - trainings_done)
                    * self.gamma * critic_values_tp1_min
                )
            )

        # Update Critics

        for critic, critic_optimizer in zip(
            self.critics, self.critic_optimizers
        ):
            critic_values_t = critic(states_t, actions_t)
            critic_losses = self.loss_function(critic_values_t, target_values)
            critic_optimizer.zero_grad()
            critic_losses.backward()

            for name, parameter in critic.named_parameters():
                if parameter.grad is not None and (
                    torch.isnan(parameter.grad).any() or
                    torch.isinf(parameter.grad).any()
                ):
                    raise ValueError(
                        f'NaN/Inf detected in DeepCritic backward pass: {name}'
                    )

            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_optimizer.step()

        # Update Actor

        pi_action_distributions_t = self.actor(states_t)
        pi_actions_t = pi_action_distributions_t.rsample()
        bounded_pi_actions_t = torch.tanh(pi_actions_t)
        jacobian_correction_t = torch.log(
            torch.clamp(1.0 - bounded_pi_actions_t.pow(2), self.epsilon)
        )
        corrected_pi_action_distributions_t = (
            pi_action_distributions_t.log_prob(
                pi_actions_t
            ) - jacobian_correction_t
        )
        pi_log_probabilities_t = (
            corrected_pi_action_distributions_t.sum(dim=-1, keepdim=True)
        )

        critic_values_t_min = None
        for critic in self.critics:
            pi_critic_values_t = critic(states_t, bounded_pi_actions_t)
            if critic_values_t_min is None:
                critic_values_t_min = pi_critic_values_t
            else:
                critic_values_t_min = (
                    torch.min(critic_values_t_min, pi_critic_values_t)
                )

        actor_losses = (
            self.alpha * pi_log_probabilities_t - critic_values_t_min
        ).mean()
        self.actor_optimizer.zero_grad()
        actor_losses.backward()

        for name, parameter in self.actor.named_parameters():
            if parameter.grad is not None and (
                torch.isnan(parameter.grad).any() or
                torch.isinf(parameter.grad).any()
            ):
                raise ValueError(
                    f'NaN/Inf detected in DeepActor backward pass: {name}'
                )

        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Update target critics with Polyak Averaging
        with torch.no_grad():
            for critic, target_critic in zip(
                    self.critics, self.target_critics
            ):
                for parameter, target_parameter in zip(
                    critic.parameters(), target_critic.parameters()
                ):
                    target_parameter.data.mul_(
                        1.0 - self.tau
                    ).add_(parameter, alpha=self.tau)
