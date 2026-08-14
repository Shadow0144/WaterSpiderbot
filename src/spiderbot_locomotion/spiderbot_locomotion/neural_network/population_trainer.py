"""Population-based evolutionary selection."""

from datetime import datetime

from .checkpoint_file_manager import CheckpointFileManager
from .deep_actor_critic_policy import DeepActorCriticPolicy


class PopulationTrainer():
    """Select the best-performing candidates to seed next training rounds."""

    class CandidateRecord:
        """Record of the candidate's filename and the total of its rewards."""

        def __init__(self, filename, epoch_reward):
            """Initialize internal state."""
            self.filename = filename
            self.epoch_reward = epoch_reward

    def __init__(self,
                 logger,
                 episodes_per_epoch=10,
                 population_size=10):
        """Initialize the class."""
        self.logger = logger
        self.episodes_per_epoch = episodes_per_epoch
        self.population_size = population_size

        self.policy = DeepActorCriticPolicy(self.logger)

        self.checkpoint_file_manager = CheckpointFileManager()

        self.current_episode = 0

        self.time_to_reach_target_s = 0.0
        self.target = None

        self.current_parent_filename = None
        self.candidate_records = []

    def get_population_checkpoint_exists(self, filename='checkpoint.csv'):
        """Get if the population checkpoint file exists."""
        return self.checkpoint_file_manager.get_population_checkpoint_exists(
            filename
        )

    def save_current_candidate_weights(self):
        """Save the current candidate weights."""
        try:
            current_candidate_filename = self.get_current_candidate_filename()
            if current_candidate_filename is not None:
                self.checkpoint_file_manager.save_weights(
                    current_candidate_filename,
                    self.policy.actor_critic,
                    self.policy.optimizer
                )
                self.logger.info(
                    f'Saved weights: {current_candidate_filename}'
                )
        except RuntimeError:
            self.logger.warn('Failed to save population')

    def save_population_checkpoint(self, filename='checkpoint.csv'):
        """Save the current state of the population training."""
        self.logger.info(f'Saving population: {filename}')
        self.save_current_candidate_weights()
        try:
            self.checkpoint_file_manager.save_population_checkpoint(
                filename,
                self.candidate_records,
                self.current_episode,
                self.current_parent_filename
            )
            self.logger.info(f'Saved population: {filename}')
        except RuntimeError:
            self.logger.warn('Failed to save population')

    def load_population_checkpoint(self, filename='checkpoint.csv'):
        """Load a population training state."""
        try:
            if not self.get_population_checkpoint_exists(filename):
                return  # Return early if no checkpoint exists

            self.logger.info(f'Loading population: {filename}')
            (
                raw_candidates,
                current_episode,
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
                self.generate_next_candidate()

            # The current episode will be incremented immediately so subtract 1
            self.current_episode = current_episode - 1
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

    def get_current_candidate_filename(self):
        """Return the current candidate's filename."""
        if self.candidate_records:
            return self.candidate_records[-1].filename
        else:
            return None

    def get_current_candidate_epoch_reward(self):
        """Return the current candidate's epoch reward."""
        if self.candidate_records:
            return self.candidate_records[-1].epoch_reward
        else:
            return None

    def set_target(self, time_to_reach_target_s, target):
        """Set the target and the estimated time to reach it."""
        self.time_to_reach_target_s = time_to_reach_target_s
        self.target = target
        self.policy.set_target(time_to_reach_target_s, target)

    def train_step(self, spiderbot_pose_msg, delta_time):
        """Perform a single training step."""
        return self.policy.train_step(spiderbot_pose_msg, delta_time)

    def get_episode_reward(self):
        """Get the episode reward from the policy and return it."""
        return self.policy.get_episode_reward()

    def add_episode_reward_to_current_epoch(self):
        """Add the episode reward to the current epoch reward."""
        if self.candidate_records:
            self.candidate_records[-1].epoch_reward += (
                self.get_episode_reward()
            )

    def start_new_training_episode(self):
        """Start another training episode or move to the next candidate."""
        self.add_episode_reward_to_current_epoch()
        if (
            not self.candidate_records or
            self.current_episode >= self.episodes_per_epoch
        ):
            self.generate_next_candidate()
        self.policy.start_new_training_episode()
        self.current_episode += 1
        self.logger.info(f'Current training episode: {self.current_episode}')

    def create_candidate_filename(self):
        """Create a candidate filename from the system clock."""
        candidate_filename = (
            f"candidate_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pt"
        )
        return candidate_filename

    def generate_next_candidate(self):
        """Create a new candidate or a new population."""
        self.save_current_candidate_weights()

        # Check if we have enough candidates to advance the population
        self.current_episode = 0
        if len(self.candidate_records) >= self.population_size:
            self.generate_next_generation()

        self.candidate_records.append(
            self.CandidateRecord(
                self.create_candidate_filename(), 0.0
            )
        )
        self.logger.info(
            f'Current candidate: {self.candidate_records[-1].filename}'
        )

        # Reset the weights back to the parent's weights
        if self.current_parent_filename is not None:
            self.policy.load_weights(self.current_parent_filename)

    def generate_next_generation(self):
        """Select the best member of the population and reseed using that."""
        # Find the candidate with the highest epoch reward to be the
        # parent of the next generation
        self.current_parent_filename = None
        highest_candidate_filename = 'None'
        if self.candidate_records:
            highest_candidate_filename = (
                self.candidate_records[0].filename
            )
            highest_epoch_reward = (
                self.candidate_records[0].epoch_reward
            )
            for candidate in self.candidate_records:
                if candidate.epoch_reward > highest_epoch_reward:
                    highest_candidate_filename = candidate.filename
                    highest_epoch_reward = candidate.epoch_reward
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
