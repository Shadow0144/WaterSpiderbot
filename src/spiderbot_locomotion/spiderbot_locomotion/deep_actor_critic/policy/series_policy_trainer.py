"""Series policy training class with population-based selection."""

from .abstract_policy_trainer import AbstractPolicyTrainer


class SeriesPolicyTrainer(AbstractPolicyTrainer):
    """Train policies and select the best-performing candidates as seeds."""

    def __init__(
            self,
            logger,
            candidate_topic_prefixes,
            num_episodes_per_candidate,
            use_soft_actor_critics_policy,
            autosave_checkpoints=True
            ):
        """Initialize the class."""
        super().__init__(
            logger,
            candidate_topic_prefixes,
            num_episodes_per_candidate,
            use_soft_actor_critics_policy,
            autosave_checkpoints
        )

        self.candidate_index = 0

    def get_are_poses_normalized(self):
        """Return if the angles are pre-scaled or require scaling."""
        candidate = self._get_candidate(None)
        if candidate:
            return candidate.get_are_poses_normalized()
        else:
            return False

    def _get_candidate(self, topic_prefix):
        """Get the current candidate if there is one."""
        return self.candidates[
            self.candidate_topic_prefixes[self.candidate_index]
        ]

    def start_new_training_episode(self):
        """Start another training episode or move to the next candidate."""
        self.episode_number += 1
        self.episode_reward = 0.0

        if (
            self.num_candidates_per_generation > 1 and
            self.episode_number > self.num_episodes_per_candidate
        ):
            self._move_to_next_candidate()

        candidate = self._get_candidate(None)
        if candidate:
            candidate.start_new_training_episode()
        else:
            self.logger.error('Failed to start training episode')
            return

        if self.num_candidates_per_generation > 1:
            self.logger.info(f'Starting training episode '
                             f'({self.episode_number}/'
                             f'{self.num_episodes_per_candidate})')
        else:
            self.logger.info(f'Starting training episode '
                             f'({self.episode_number})')

        # Save a checkpoint
        if self.autosave_checkpoints:
            self.save_population_checkpoint()

    def _move_to_next_candidate(self):
        """Create a new candidate or a new population."""
        self.episode_number = 1
        self.candidate_index += 1

        # Check if we have enough candidates to advance the population
        if self.candidate_index >= self.num_candidates_per_generation:
            self._generate_next_generation()
            self.logger.info(f'Starting new generation '
                             f'({self.generation_number})')

        self.logger.info(f'Training candidate '
                         f'({self.candidate_index + 1}/'
                         f'{self.num_candidates_per_generation})')
