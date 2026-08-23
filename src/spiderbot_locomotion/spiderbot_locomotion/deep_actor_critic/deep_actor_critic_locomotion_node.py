"""Spiderbot locomotion node using a deep learning actor-critic."""

from std_msgs.msg import Float64

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

        self.current_step_reward_publisher = self.create_publisher(
            Float64, 'current_step_reward', 10)

        self.episode_reward_publisher = self.create_publisher(
            Float64, 'episode_reward', 10)

        self.reset_learned_weights_service = self.create_service(
            Trigger,
            'reset_learned_weights',
            self.reset_learned_weights_callback
        )

    def publish_current_step_reward(self, reward):
        """Publish the reward for the current step."""
        msg = Float64()
        msg.data = reward
        self.current_step_reward_publisher.publish(msg)

    def publish_episode_reward(self, reward):
        """Publish the reward for the full training episode."""
        msg = Float64()
        msg.data = reward
        self.episode_reward_publisher.publish(msg)

    def reset_learned_weights_callback(self, request, response):
        """Backup the current weights and start with new random weights."""
        self.locomotion_module.reset_learned_weights()
        response.success = True
        response.message = 'Success'
        return response
