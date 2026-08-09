"""Provide a description for a Spiderbot."""

from rclpy.node import Node

from spiderbot_interfaces.msg import LegDescription
from spiderbot_interfaces.srv import GetSpiderbotDescription

from .spiderbot import Spiderbot


class DescriptionNode(Node):
    """A description node for a Spiderbot."""

    def __init__(self):
        """Initialize and describe a Spiderbot."""
        super().__init__('description_node')

        self.get_logger().info('Starting spiderbot description node')

        self.spider = Spiderbot()

        self.get_spiderbot_description_service = self.create_service(
            GetSpiderbotDescription,
            'get_spiderbot_description',
            self.get_spiderbot_description_callback
        )

        self.get_logger().info('Spiderbot description node started')

    def is_running(self):
        """Return if the node is running or if it's ready to shut down."""
        return True

    def get_spiderbot_description_callback(self, request, response):
        """Publish the Spiderbot description."""
        leg_descriptions = []
        for i, leg_name in enumerate(self.spider.leg_names):
            leg_description = LegDescription()
            leg_description.leg_name = leg_name
            leg_description.segment_lengths = (
                self.spider.segment_lengths_per_leg[i]
            )
            leg_descriptions.append(leg_description)
        response.leg_descriptions = leg_descriptions
        response.spec_xml = self.spider.spec.to_xml()
        return response
