"""Provide a description for a Spiderbot."""

from rclpy.node import Node

from spiderbot_interfaces.msg import LegDescription
from spiderbot_interfaces.srv import GetSpiderbotDescription

from .spiderbot import Spiderbot


def convert_vector3_to_list(vector3):
    """Convert a Vector3 object to a list."""
    return (vector3.x, vector3.y, vector3.z)


class DescriptionNode(Node):
    """A description node for a Spiderbot."""

    def __init__(self):
        """Initialize and describe a Spiderbot."""
        super().__init__('description_node')

        self.spider = Spiderbot()

        self.get_spiderbot_description_service = self.create_service(
            GetSpiderbotDescription,
            'get_spiderbot_description',
            self.get_spiderbot_description_callback
        )

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
