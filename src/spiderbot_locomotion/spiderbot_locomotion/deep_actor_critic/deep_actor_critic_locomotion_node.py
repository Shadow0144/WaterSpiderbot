"""Spiderbot locomotion node using a deep learning actor-critic."""


from spiderbot_interfaces.msg import TrainingStatus

from std_srvs.srv import Trigger

from .deep_actor_critic_module import DeepActorCriticModule
from ..locomotion_node import LocomotionNode


class DeepActorCriticLocomotionNode(LocomotionNode):
    """Spiderbot locomotion using a deep learning actor-critic."""

    def __init__(self):
        """Initialize and run a Spiderbot locomotor."""
        super().__init__('deep_actor_critic_locomotion_node')

        # Set the module after getting the description
        self.locomotion_module = DeepActorCriticModule(
            self,
            self.spiderbot_description
        )

        self.training_status_publisher = self.create_publisher(
            TrainingStatus,
            'training_status',
            10
        )

        self.reset_learned_weights_service = self.create_service(
            Trigger,
            'reset_learned_weights',
            self.reset_learned_weights_callback
        )

    def publish_training_status(self, training_status):
        """Publish information on the training and the reward."""
        msg = TrainingStatus()
        msg.step_reward = training_status.step_reward
        msg.episode_reward = training_status.episode_reward
        msg.candidate_reward = training_status.candidate_reward
        msg.using_population_training = (
            training_status.using_population_training
        )
        msg.episode_number = training_status.episode_number
        msg.episodes_per_candidate = training_status.episodes_per_candidate
        msg.candidate_number = training_status.candidate_number
        msg.candidates_per_generation = (
            training_status.candidates_per_generation
        )
        msg.generation_number = training_status.generation_number
        msg.target_reached = training_status.target_reached
        msg.episode_terminated = training_status.episode_terminated
        msg.reward_component_labels = (
            training_status.reward_component_labels
        )
        msg.reward_component_values = (
            training_status.reward_component_values
        )
        self.training_status_publisher.publish(msg)

    def reset_learned_weights_callback(self, request, response):
        """Backup the current weights and start with new random weights."""
        self.locomotion_module.reset_learned_weights()
        response.success = True
        response.message = 'Success'
        return response
