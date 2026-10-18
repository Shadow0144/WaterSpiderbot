"""Parallel policy training class with population-based selection."""


from .abstract_policy_trainer import AbstractPolicyTrainer


class ParallelPolicyTrainer(AbstractPolicyTrainer):
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

    def get_are_actions_normalized(self):
        """Return if the angles are pre-scaled or require scaling."""
        normalized = False
        for candidate in self.candidates.values():
            normalized &= candidate.get_are_actions_normalized()
        return normalized

    def _get_candidate(self, topic_prefix):
        """Get the topic candidate if there is one."""
        if topic_prefix in self.candidates.keys():
            return self.candidates[topic_prefix]
        else:
            return None

    def start_new_training_episode(self):
        """Start another training episode or move to the next candidate."""
        self.episode_number += 1
        self.episode_reward = 0.0

        if self.episode_number > self.num_episodes_per_candidate:
            self._generate_next_generation()

        for candidate in self.candidates.values():
            candidate.start_new_training_episode()

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
