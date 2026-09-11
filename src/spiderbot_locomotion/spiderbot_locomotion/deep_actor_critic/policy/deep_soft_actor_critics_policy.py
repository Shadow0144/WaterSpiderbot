"""Locomotion policy based on a Soft Actor-Critics Deep Neural Network."""

import copy
import random
from collections import deque

import torch
from torch import nn

from .deep_actor_critic_policy import DeepActorCriticPolicy
from ..neural_network.deep_actor import DeepActor
from ..neural_network.deep_critic import DeepCritic
from ..neural_network.step_transition import StepTransition
from ..neural_network.step_transition import TrainingObservation
from ..utility import construct_input_vector


class DeepSoftActorCriticsPolicy(DeepActorCriticPolicy):
    """Provides an action based on the current state."""

    def __init__(self, logger):
        """Initialize the locomotion neural network."""
        super().__init__(logger)

        # Output angles will need to be scaled up
        self.angles_scaled = False

        self.batch_size = 100
        self.buffer_size = 10_000
        self.training_observations = deque(maxlen=self.buffer_size)

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
        super()._determine_layer_sizes()
        self.num_frames_k = 5  # Number of frames in each state
        self.num_actor_inputs_k = self.num_actor_inputs * self.num_frames_k
        self.num_critic_inputs = (
            self.num_actor_inputs_k + self.num_actor_outputs
        )

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

    def _reset_frame_queue(self):
        """Zero out all the observed frames in the buffer."""
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
        self._reset_frame_queue()
        self.transition_t = None
        self.reward_function.start_new_training_episode()

    def _construct_state(self, spiderbot_pose):
        """Construct a state vector from the latest observation."""
        frame_t = construct_input_vector(
                self.target,
                spiderbot_pose,
                self.device
        )
        self.frames.append(frame_t)
        state_t = torch.cat(list(self.frames), dim=-1)
        return state_t

    def select_action(self, spiderbot_pose, deterministic=False):
        """Select the next action."""
        if self.target is None:
            return None  # Exit early if there is no target

        state_t = self._construct_state(spiderbot_pose)

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

        transition_t = StepTransition(
            state_t,
            action_t,
            log_probability_t
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

        state_t = self._construct_state(spiderbot_pose)

        if self.transition_t is not None:
            # If there was a previous state,
            # calculate the reward and train the actor-critic
            reward_t, training_done = (
                self.reward_function.compute_step_reward(
                    self.target,
                    spiderbot_pose,
                    delta_time
                )
            )

            training_observation = TrainingObservation(
                state_t=self.transition_t.state_t,
                action_t=self.transition_t.action_t,
                log_probability_t=self.transition_t.log_probability_t,
                state_tp1=state_t,
                reward_t=reward_t,
                training_done=training_done
            )
            self.training_observations.append(training_observation)

            if len(self.training_observations) >= self.buffer_size:
                self._train_actor_critic_step()

        action_t, self.transition_t = (
            self._select_action_stochastic(
                state_t
            )
        )

        return action_t, reward_t, training_done

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
            [observation.training_done for observation in batch_observations],
            dtype=torch.float32, device=self.device).unsqueeze(1)

        # Compute Bellman Target (t+1)

        with torch.no_grad():
            action_distributions_tp1 = self.actor(states_tp1)
            actions_tp1 = action_distributions_tp1.sample()
            log_probability_tp1 = action_distributions_tp1.log_prob(
                actions_tp1
            ).sum(dim=-1, keepdim=True)

            critic_values_tp1_min = None
            for target_critic in self.target_critics:
                critic_values_tp1 = target_critic(states_tp1, actions_tp1)
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
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_optimizer.step()

        # Update Actor

        pi_action_distributions_t = self.actor(states_t)
        pi_actions_t = pi_action_distributions_t.rsample()
        bounded_pi_actions_t = torch.tanh(pi_actions_t)
        jacobian_correction_t = torch.log(
            1.0 - bounded_pi_actions_t.pow(2) + self.epsilon
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
            pi_critic_values_t = critic(states_t, pi_actions_t)
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
