"""Overlay for displaying information about a scene."""

import mujoco


class SceneOverlay():
    """Overlay."""

    def __init__(
            self,
            topic_prefix,
            train_parallel
            ):
        """Initialize the overlay."""
        self.topic_prefix = topic_prefix
        self.train_parallel = train_parallel

        self.displaying_target_number = False
        self.target_number = 0
        self.max_target_number = 0
        self.displaying_step_reward = False
        self.step_reward = 0.0
        self.displaying_episode_reward = False
        self.episode_number = 0
        self.episode_reward = 0.0
        self.displaying_candidate_reward = False
        self.candidate_number = 0
        self.candidate_reward = 0.0
        self.generation_number = 0

        self.display_num_limits = False
        self.num_targets_per_episode = 0
        self.num_episodes_per_candidate = 0
        self.num_candidates_per_generation = 0

        self.reward_component_labels = None
        self.reward_component_values = None

    def set_training_configuration(
            self,
            num_targets_per_episode,
            num_episodes_per_candidate,
            num_candidates_per_generation
            ):
        """Set the training configuration for the display."""
        self.display_num_limits = True
        self.num_targets_per_episode = num_targets_per_episode
        self.num_episodes_per_candidate = num_episodes_per_candidate
        self.num_candidates_per_generation = num_candidates_per_generation

    def update_target_number(self, target_number):
        """Enable displaying the target number and update it."""
        self.displaying_target_number = True
        self.target_number = target_number

    def update_training_status(self, training_status_msg):
        """Enable displaying the step reward and update it."""
        self.displaying_step_reward = True
        self.step_reward = training_status_msg.step_reward
        self.displaying_episode_reward = True
        self.episode_number = training_status_msg.episode_number
        self.episode_reward = training_status_msg.episode_reward
        if training_status_msg.using_population_training:
            self.displaying_candidate_reward = True
            self.candidate_number = training_status_msg.candidate_number
            self.candidate_reward = training_status_msg.candidate_reward
            self.generation_number = training_status_msg.generation_number
        self.reward_component_labels = (
            training_status_msg.reward_component_labels
        )
        self.reward_component_values = (
            training_status_msg.reward_component_values
        )

    def add_overlays(
            self,
            viewer_context,
            viewport
            ):
        """Add the training overlays to the viewport."""
        left_labels = []
        left_values = []

        if self.displaying_target_number:
            left_labels.append('Target number:')
            if not self.display_num_limits:
                left_values.append(f'{self.target_number}')
            else:
                left_values.append(f'{self.target_number} / '
                                   f'{self.num_targets_per_episode}')

        if self.displaying_step_reward:
            left_labels.append('Step reward:')
            left_values.append(f'{self.step_reward:.3f}')

        if self.displaying_episode_reward:
            left_labels.append('Episode number:')
            if not self.display_num_limits:
                left_values.append(f'{self.episode_number}')
            else:
                left_values.append(f'{self.episode_number} / '
                                   f'{self.num_episodes_per_candidate}')
            left_labels.append('Episode reward:')
            left_values.append(f'{self.episode_reward:.3f}')

        if self.displaying_candidate_reward:
            left_labels.append('Candidate number:')
            if not self.display_num_limits:
                left_values.append(f'{self.candidate_number}')
            else:
                left_values.append(f'{self.candidate_number} / '
                                   f'{self.num_candidates_per_generation}')
            left_labels.append('Candidate reward:')
            left_values.append(f'{self.candidate_reward:.3f}')
            left_labels.append('Generation number:')
            left_values.append(f'{self.generation_number}')

        if left_labels and left_values:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                '\n'.join(left_labels),
                '\n'.join(left_values),
                viewer_context
            )

        if self.reward_component_labels and self.reward_component_values:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                viewport,
                '\n'.join(self.reward_component_labels),
                '\n'.join(
                    [f'{component_value:.3f}' for component_value in
                        self.reward_component_values]
                ),
                viewer_context
            )

        if self.train_parallel:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOP,
                viewport,
                self.topic_prefix,
                None,
                viewer_context
            )
