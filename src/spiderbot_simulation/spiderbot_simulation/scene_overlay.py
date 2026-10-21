"""Overlay for displaying information about a scene."""

import mujoco


class SceneOverlay():
    """Overlay."""

    def __init__(
            self,
            viewer_context,
            viewport,
            topic_prefix,
            train_parallel
            ):
        """Initialize the overlay."""
        self.viewer_context = viewer_context
        self.viewport = viewport
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

        self.target_reached = False
        self.episode_terminated = False

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
        component_values = training_status_msg.reward_component_values
        # Convert to strings with 3 decimal places
        self.reward_component_values = (
            [f'{component_value:.3f}'
             for component_value in
             component_values]
        )
        self.target_reached = training_status_msg.target_reached
        self.episode_terminated = training_status_msg.episode_terminated

    def add_overlays(self, viewer_config):
        """Add the viewer settings and training overlays to the viewport."""
        keybinds = {
            '[<-] Previous Candidate': '',
            '[->] Next Candidate': '',
            '[f] Toggle Camera Following': '',
            '[r] Reset Camera Tranforms': '',
            '[c] Camera Views Visible:': (
                'Enabled'
                if viewer_config['camera_views_visible'] else
                'Disabled'
            ),
            '[j] Joints Visible:': (
                'Enabled'
                if viewer_config['joints_visible'] else
                'Disabled'
            ),
            '[v] Frames Visible:': (
                'Enabled'
                if viewer_config['frames_visible'] else
                'Disabled'
            ),
            '[t] Translucency:': (
                'Enabled'
                if viewer_config['translucency'] else
                'Disabled'
            ),
        }
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            self.viewport,
            '\n'.join(keybinds.keys()),
            '\n'.join(
                [f'{setting}' for setting in keybinds.values()]
            ),
            self.viewer_context
        )

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
                self.viewport,
                '\n'.join(left_labels),
                '\n'.join(left_values),
                self.viewer_context
            )

        if self.reward_component_labels and self.reward_component_values:
            reward_component_labels = self.reward_component_labels.copy()
            reward_component_values = self.reward_component_values.copy()

            reward_component_labels.append('Target reached:')
            reward_component_values.append(f'{self.target_reached}')
            reward_component_labels.append('Episode terminated:')
            reward_component_values.append(f'{self.episode_terminated}')

            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                self.viewport,
                '\n'.join(reward_component_labels),
                '\n'.join(reward_component_values),
                self.viewer_context
            )

        if self.train_parallel:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOP,
                self.viewport,
                self.topic_prefix,
                None,
                self.viewer_context
            )
