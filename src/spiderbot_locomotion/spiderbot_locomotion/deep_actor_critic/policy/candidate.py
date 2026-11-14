"""Training candidate."""

from .deep_soft_actor_critics_policy import DeepSoftActorCriticsPolicy


class Candidate:
    """Record of the candidate's topic_prefix, filename, and total reward."""

    def __init__(
            self,
            logger,
            topic_prefix,
            filename,
            candidate_reward,
            candidate_number
            ):
        """Initialize internal state."""
        self.logger = logger
        self.topic_prefix = topic_prefix
        self.filename = filename
        self.candidate_reward = candidate_reward
        self.policy = DeepSoftActorCriticsPolicy(
            self.logger,
            self.filename,
            False
        )
        self.candidate_number = candidate_number
        self.episode_terminated = False

    def get_are_actions_normalized(self):
        """Return if the angles are pre-scaled or require scaling."""
        if self.policy:
            return self.policy.get_are_actions_normalized()
        else:
            return False

    def get_model_weights_file_exists(self):
        """Return if the weight file exists for this candidate."""
        try:
            if self.policy:
                return self.policy.get_model_weights_file_exists()
        except RuntimeError as e:
            self.logger.error(f'Failed to check model weights file: {e}')

    def save_model_weights(self, filename_override=None):
        """Save the model weights."""
        try:
            if self.policy:
                self.policy.save_model_weights(filename_override)
        except RuntimeError as e:
            self.logger.error(f'Failed to save model weights: {e}')

    def load_model_weights(self, filename_override=None):
        """Load the model weights."""
        try:
            if self.policy:
                self.policy.load_model_weights(filename_override)
        except (RuntimeError, FileNotFoundError) as e:
            self.logger.error(f'Failed to load model weights: {e}')

    def delete_model_weights(self):
        """Delete the model weights file."""
        try:
            if self.policy:
                self.policy.delete_saved_weights()
        except (RuntimeError, FileNotFoundError) as e:
            self.logger.error(f'Failed to delete model weights: {e}')

    def start_new_training_episode(self):
        """Start another training episode."""
        if self.policy:
            self.policy.start_new_training_episode()
        self.episode_terminated = False

    def set_target(self, target):
        """Set the target for the policy to approach."""
        if self.policy:
            self.policy.set_target(target)

    def train_step(self, spiderbot_pose_msg, delta_time):
        """Perform a single training step."""
        if self.policy and not self.episode_terminated:
            action_t, training_status = self.policy.train_step(
                spiderbot_pose_msg,
                delta_time
            )
            self.episode_terminated = training_status.episode_terminated
            return action_t, training_status
        else:
            return None, None
