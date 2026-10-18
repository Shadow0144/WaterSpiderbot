"""Provide a description for a Spiderbot."""

from rclpy.node import Node

from spiderbot_interfaces.msg import LegDescription
from spiderbot_interfaces.srv import GetSpiderbotDescription
from spiderbot_interfaces.srv import GetWorldDescription

from .spiderbot import Spiderbot
from .world import World


class DescriptionNode(Node):
    """A description node for a Spiderbot."""

    def __init__(self):
        """Initialize and describe a Spiderbot."""
        super().__init__('description_node')

        self.get_logger().info('Starting spiderbot description node')

        self.spiderbot = Spiderbot()

        self.world = World()

        self.get_spiderbot_description_service = self.create_service(
            GetSpiderbotDescription,
            'get_spiderbot_description',
            self.get_spiderbot_description_callback
        )

        self.get_world_description_service = self.create_service(
            GetWorldDescription,
            'get_world_description',
            self.get_world_description_callback
        )

        self.get_logger().info('Spiderbot description node started')

    def is_running(self):
        """Return if the node is running or if it's ready to shut down."""
        return True

    def get_spiderbot_description_callback(self, request, response):
        """Publish the Spiderbot description."""
        leg_descriptions = []
        for i, leg_name in enumerate(self.spiderbot.leg_names):
            leg_description = LegDescription()
            leg_description.leg_name = leg_name
            leg_description.segment_lengths = (
                self.spiderbot.segment_lengths_per_leg[i]
            )
            leg_descriptions.append(leg_description)
        response.leg_descriptions = leg_descriptions
        response.mesh_dir = self.spiderbot.get_mesh_path()
        response.texture_dir = self.spiderbot.get_texture_path()
        response.spec_xml = self.spiderbot.get_spec_xml()
        return response

    def get_world_description_callback(self, request, response):
        """Publish the world description."""
        response.mesh_dir = self.world.get_mesh_path()
        response.texture_dir = self.world.get_texture_path()
        response.spec_xml = self.world.get_spec_xml()
        return response
