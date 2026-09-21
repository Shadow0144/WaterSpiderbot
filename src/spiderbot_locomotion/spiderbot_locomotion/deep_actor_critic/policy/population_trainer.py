"""Population-based evolutionary selection."""

from datetime import datetime

from .deep_actor_critic_policy import DeepActorCriticPolicy
from .deep_soft_actor_critics_policy import DeepSoftActorCriticsPolicy
from ..neural_network.checkpoint_file_manager import CheckpointFileManager


class PopulationTrainer():
    """Select the best-performing candidates to seed next training rounds."""

    class CandidateRecord:
        """Record of the candidate's filename and the total of its rewards."""

        def __init__(self, filename, candidate_reward):
            """Initialize internal state."""
            self.filename = filename
            self.candidate_reward = candidate_reward

    def __init__(self,
                 logger,
                 episodes_per_candidate=10,
                 candidates_per_generation=10,
                 use_soft_actor_critics_policy=True):
        """Initialize the class."""
        self.logger = logger
        self.episodes_per_candidate = episodes_per_candidate
        self.candidates_per_generation = candidates_per_generation

        if use_soft_actor_critics_policy:
            self.policy = DeepSoftActorCriticsPolicy(self.logger)
        else:
            self.policy = DeepActorCriticPolicy(self.logger)

        self.checkpoint_file_manager = CheckpointFileManager()

        self.episode_number = 0
        self.candidate_number = 0
        self.generation_number = 1

        self.target = None

        self.current_parent_filename = None
        self.candidate_records = []

    def get_poses_normalized(self):
        """Return if the angles are pre-scaled or require scaling."""
        return self.policy.get_poses_normalized()

    def get_population_checkpoint_exists(self, filename='checkpoint.csv'):
        """Get if the population checkpoint file exists."""
        return self.checkpoint_file_manager.get_population_checkpoint_exists(
            filename
        )

    def save_current_candidate_weights(self):
        """Save the current candidate weights."""
        try:
            current_candidate_filename = self._get_current_candidate_filename()
            if current_candidate_filename is not None:
                self.policy.save_weights(current_candidate_filename)
                self.logger.info(
                    f'Saved weights: {current_candidate_filename}'
                )
        except RuntimeError as e:
            self.logger.warn(f'Failed to save population: {e}')

    def save_population_checkpoint(self, filename='checkpoint.csv'):
        """Save the current state of the population training."""
        self.logger.info(f'Saving population: {filename}')
        self.save_current_candidate_weights()
        try:
            self.checkpoint_file_manager.save_population_checkpoint(
                filename,
                self.candidate_records,
                self.episode_number,
                self.candidate_number,
                self.generation_number,
                self.current_parent_filename
            )
            self.logger.info(f'Saved population: {filename}')
        except RuntimeError as e:
            self.logger.warn(f'Failed to save population: {e}')

    def load_population_checkpoint(self, filename='checkpoint.csv'):
        """Load a population training state."""
        try:
            self.logger.info(f'Loading population: {filename}')
            if not self.get_population_checkpoint_exists(filename):
                self.logger.warning(
                    f'Population checkpoint file {filename} not found'
                )
                return  # Return early if no checkpoint exists

            (
                raw_candidates,
                episode_number,
                self.candidate_number,
                self.generation_number,
                parent_candidate_filename
            ) = (
                self.checkpoint_file_manager.load_population_checkpoint(
                    filename
                )
            )

            self.candidate_records = []
            for raw_candidate in raw_candidates:
                candidate = self.CandidateRecord(raw_candidate[0],
                                                 raw_candidate[1])
                self.candidate_records.append(candidate)
            if not self.candidate_records:
                self._generate_next_candidate()

            # The current episode will be incremented immediately so subtract 1
            self.episode_number = episode_number - 1
            self.policy.episode_number = self.episode_number
            if parent_candidate_filename:
                self.current_parent_filename = parent_candidate_filename
                self.policy.load_weights(parent_candidate_filename)
                self.logger.info(
                    f'Loaded parent weights: {parent_candidate_filename}'
                )
            else:
                self.current_parent_filename = None
                candidate_filename = self.candidate_records[-1].filename
                if self.checkpoint_file_manager.get_model_weights_exists(
                    candidate_filename
                ):
                    self.policy.load_weights(candidate_filename)
                    self.logger.info(
                        f'Loaded candidate weights: {candidate_filename}'
                    )

            self.logger.info(f'Loaded population: {filename}')
        except (RuntimeError, FileNotFoundError):
            self.logger.warn('Failed to load population')

    def delete_population_checkpoint(self, filename):
        """Delete a population checkpoint file."""
        if self.checkpoint_file_manager.get_population_checkpoint_exists(
            filename
        ):
            self.checkpoint_file_manager.delete_population_checkpoint(
                filename
            )
            # TODO: Delete the candidate files too
            self.logger.info(f'Deleted population: {filename}')

    def _get_current_candidate_filename(self):
        """Return the current candidate's filename."""
        if self.candidate_records:
            return self.candidate_records[-1].filename
        else:
            return None

    def get_current_candidate_candidate_reward(self):
        """Return the current candidate's candidate reward."""
        if self.candidate_records:
            return self.candidate_records[-1].candidate_reward
        else:
            return None

    def set_target(self, target):
        """Set the target and the estimated time to reach it."""
        self.target = target
        self.policy.set_target(self.target)

    def train_step(self, spiderbot_pose_msg, delta_time):
        """Perform a single training step."""
        action_t, reward_information = (
            self.policy.train_step(spiderbot_pose_msg, delta_time)
        )

        # Add the reward to the candidate
        candidate_reward = 0.0
        candidate_number = 0
        if self.candidate_records:
            self.candidate_records[-1].candidate_reward += (
                reward_information.step_reward
            )
            candidate_reward = self.candidate_records[-1].candidate_reward
            candidate_number = len(self.candidate_records)

        reward_information.candidate_reward = candidate_reward
        reward_information.episodes_per_candidate = self.episodes_per_candidate
        reward_information.candidate_number = candidate_number
        reward_information.candidates_per_generation = (
            self.candidates_per_generation
        )
        reward_information.generation_number = self.generation_number

        return action_t, reward_information

    def start_new_training_episode(self):
        """Start another training episode or move to the next candidate."""
        if (
            not self.candidate_records or
            self.episode_number >= self.episodes_per_candidate
        ):
            self._generate_next_candidate()
        self.policy.start_new_training_episode(False)
        self.episode_number += 1
        self.logger.info(f'Starting training episode '
                         f'{self.episode_number}/'
                         f'{self.episodes_per_candidate}')

    def _create_candidate_filename(self):
        """Create a candidate filename from the system clock."""
        candidate_filename = (
            f"candidate_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pt"
        )
        return candidate_filename

    def _generate_next_candidate(self):
        """Create a new candidate or a new population."""
        self.save_current_candidate_weights()

        # Check if we have enough candidates to advance the population
        self.episode_number = 0
        self.candidate_number = len(self.candidate_records)
        self.policy.reset()
        if self.candidate_number >= self.candidates_per_generation:
            self._generate_next_generation()
            self.logger.info(f'Starting generation '
                             f'{self.generation_number}')

        self.candidate_records.append(
            self.CandidateRecord(
                self._create_candidate_filename(), 0.0
            )
        )
        self.logger.info(
            f'Current candidate: {self.candidate_records[-1].filename}'
        )

        # Reset the weights back to the parent's weights
        if self.current_parent_filename is not None:
            self.policy.load_weights(self.current_parent_filename)

        self.logger.info(f'Starting training candidate '
                         f'{self.candidate_number}/'
                         f'{self.candidates_per_generation}')

    def _generate_next_generation(self):
        """Select the best member of the population and reseed using that."""
        # Find the candidate with the highest candidate reward to be the
        # parent of the next generation
        self.episode_number = 0
        self.policy.reset()
        self.candidate_number = 0
        self.generation_number += 1
        self.current_parent_filename = None
        highest_candidate_filename = 'None'
        if self.candidate_records:
            highest_candidate_filename = (
                self.candidate_records[0].filename
            )
            highest_candidate_reward = (
                self.candidate_records[0].candidate_reward
            )
            for candidate in self.candidate_records:
                if candidate.candidate_reward > highest_candidate_reward:
                    highest_candidate_filename = candidate.filename
                    highest_candidate_reward = candidate.candidate_reward
            self.current_parent_filename = highest_candidate_filename
            self.policy.load_weights(self.current_parent_filename)

        # Start over with a new empty population
        previous_candidate_records = self.candidate_records
        self.candidate_records = []

        # Delete the old candidate files
        for candidate in previous_candidate_records:
            candidate_filename = candidate.filename
            if (
                candidate_filename != self.current_parent_filename and
                self.checkpoint_file_manager.get_model_weights_exists(
                    candidate_filename
                )
            ):
                self.checkpoint_file_manager.delete_saved_weights(
                    candidate_filename
                )

        # Save a checkpoint
        self.save_population_checkpoint()

        self.logger.info(
            f'Starting next population from {self.current_parent_filename}'
        )
