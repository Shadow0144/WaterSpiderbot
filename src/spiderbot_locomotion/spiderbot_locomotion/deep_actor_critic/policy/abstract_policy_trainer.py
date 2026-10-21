"""Base policy training class with population-based selection."""

from abc import ABC, abstractmethod
from datetime import datetime

from .candidate import Candidate
from ..file_managers.checkpoint_file_manager import CheckpointFileManager


class AbstractPolicyTrainer(ABC):
    """Abstract base class for training policies."""

    def __init__(
            self,
            logger,
            candidate_topic_prefixes,
            num_episodes_per_candidate,
            use_soft_actor_critics_policy,
            autosave_checkpoints=True
            ):
        """Initialize the class."""
        self.logger = logger
        self.candidate_topic_prefixes = candidate_topic_prefixes
        self.num_episodes_per_candidate = num_episodes_per_candidate
        self.num_candidates_per_generation = len(self.candidate_topic_prefixes)
        self.autosave_checkpoints = autosave_checkpoints
        self.use_soft_actor_critics_policy = use_soft_actor_critics_policy

        self.checkpoint_file_manager = CheckpointFileManager()
        self.model_weights_filename = 'model_weights.pt'

        self.episode_number = 0
        self.candidate_index = 0
        self.generation_number = 1

        self.episode_reward = 0.0

        self.candidates = {}
        self._generate_candidates()

    def _generate_candidates(self):
        """Create all the candidates in the generation."""
        # Candidate topic prefixes determine the name and number of
        # candidates, but they will not be used in the topics themselves
        for index, topic_prefix in enumerate(self.candidate_topic_prefixes):
            self.candidates[topic_prefix] = (
                Candidate(
                    self.logger,
                    topic_prefix,
                    self._create_candidate_filename(topic_prefix),
                    0.0,
                    self.use_soft_actor_critics_policy,
                    (index + 1)
                    )
                )

    def _create_candidate_filename(self, topic_prefix):
        """Create a candidate filename from the system clock."""
        candidate_name = topic_prefix[:-1]  # Remove the '/'
        candidate_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        candidate_filename = (
            f'{candidate_name}_{candidate_time}.pt'
        )
        return candidate_filename

    @abstractmethod
    def get_are_actions_normalized(self):
        """Return if the angles are pre-scaled or require scaling."""
        pass

    def get_population_checkpoint_exists(self):
        """Get if the population checkpoint file exists."""
        return self.checkpoint_file_manager.get_population_checkpoint_exists()

    def save_population_checkpoint(self):
        """Save the current state of the population training."""
        self.logger.info('Saving population')
        for candidate in self.candidates.values():
            candidate.save_weights()
        try:
            self.checkpoint_file_manager.save_population_checkpoint(
                self.candidates.values(),
                self.episode_number,
                self.candidate_index,
                self.generation_number
            )
            self.logger.info('Saved population')
        except RuntimeError as e:
            self.logger.warning(f'Failed to save population: {e}')

    def load_population_checkpoint(self):
        """Load a population training state."""
        try:
            self.logger.info('Loading population')
            if not self.get_population_checkpoint_exists():
                self.logger.warning(
                    'Population checkpoint file not found'
                )
                return  # Return early if no checkpoint exists

            (
                candidate_filenames,
                candidate_rewards,
                episode_number,
                self.candidate_index,
                self.generation_number
            ) = (
                self.checkpoint_file_manager.load_population_checkpoint()
            )
            # The current episode will be incremented immediately so subtract 1
            self.episode_number = episode_number - 1

            if (
                len(candidate_filenames) != len(self.candidates) or
                len(candidate_rewards) != len(self.candidates)
            ):
                raise RuntimeError('Candidate count does not match')

            for c, candidate in enumerate(self.candidates.values()):
                candidate.filename = candidate_filenames[c]
                candidate.candidate_reward = candidate_rewards[c]
                candidate.load_weights()

            self.logger.info('Loaded population')
        except (RuntimeError, FileNotFoundError):
            self.logger.warning('Failed to load population')

    def delete_population_checkpoint(self):
        """Delete a population checkpoint file."""
        if self.checkpoint_file_manager.get_population_checkpoint_exists():
            self.checkpoint_file_manager.delete_population_checkpoint()
            self.logger.info('Deleted population checkpoint')
            for candidate in self.candidates.values():
                candidate.delete_weights()

    @abstractmethod
    def _get_candidate(self, topic_prefix):
        """Get the current candidate if there is one."""
        pass

    def get_policy(self, topic_prefix):
        """Get the current policy."""
        return self._get_candidate(topic_prefix).policy

    def set_target(self, topic_prefix, target):
        """Set the target for the policy to approach."""
        candidate = self._get_candidate(topic_prefix)
        if candidate:
            candidate.set_target(target)

    def train_step(self, topic_prefix, spiderbot_pose_msg, delta_time):
        """Perform a single training step."""
        candidate = self._get_candidate(topic_prefix)
        if not candidate:
            return None, None

        action_t, training_status = (
            candidate.train_step(spiderbot_pose_msg, delta_time)
        )

        if action_t is None or training_status is None:
            return None, None

        # Add the reward to the episode
        self.episode_reward += training_status.step_reward
        training_status.episode_number = self.episode_number
        training_status.episode_reward = self.episode_reward

        # Add the reward to the candidate
        candidate_reward = 0.0
        candidate.candidate_reward += (
            training_status.step_reward
        )
        candidate_reward = candidate.candidate_reward

        # Fill in more of the reward information object
        training_status.candidate_reward = candidate_reward
        training_status.num_episodes_per_candidate = (
            self.num_episodes_per_candidate
        )
        training_status.candidate_number = candidate.candidate_number
        training_status.num_candidates_per_generation = (
            self.num_candidates_per_generation
        )
        training_status.generation_number = self.generation_number

        return action_t, training_status

    @abstractmethod
    def start_new_training_episode(self):
        """Start another training episode or move to the next candidate."""
        pass

    def _generate_next_generation(self):
        """Select the best member of the population and re-seed using that."""
        # Find the candidate with the highest candidate reward to be the
        # parent of the next generation
        self.candidate_index = 0
        self.episode_number = 1
        self.generation_number += 1

        best_candidate = None
        if self.candidates:
            highest_candidate_reward = float('-inf')
            for candidate in self.candidates.values():
                if candidate.candidate_reward > highest_candidate_reward:
                    best_candidate = candidate
                    highest_candidate_reward = candidate.candidate_reward

        if best_candidate:
            self.logger.info(
                f'Starting new generation from '
                f'{best_candidate.topic_prefix} '
                f'[{best_candidate.filename}]'
            )
            best_candidate.save_weights(
                filename_override=self.model_weights_filename
            )

            # Delete the old candidate files
            for candidate in self.candidates.values():
                candidate.delete_weights()

            self._generate_candidates()
            for candidate in self.candidates.values():
                candidate.load_weights(
                    filename_override=self.model_weights_filename
                )
