"""Constructs a description of a world."""

import os

from ament_index_python.packages import get_package_share_directory

import mujoco


class World:
    """Main description of a world."""

    def __init__(self):
        """Initialize a world."""
        self._load_model()

    def get_mesh_path(self):
        """Get the path to the mesh directory from the share directory."""
        mesh_path = os.path.join(self._get_model_path(), 'meshes')
        return mesh_path

    def get_texture_path(self):
        """Get the path to the texture directory from the share directory."""
        texture_path = os.path.join(self._get_model_path(), 'textures')
        return texture_path

    def get_spec_xml(self):
        """Get the spec xml."""
        return self.spec.to_xml()

    def _get_model_path(self):
        """Get the path to the model directory from the share directory."""
        share_dir = get_package_share_directory('spiderbot_description')
        model_path = os.path.join(share_dir, 'models')
        return model_path

    def _get_world_model_file(self):
        """Get the file path to the world model file."""
        model_filepath = os.path.join(
            self._get_model_path(),
            'test_world.xml'
        )
        return model_filepath

    def _load_model(self):
        """
        Load and finish the model and data.

        Loads the Spiderbot partial description from a file
        and adds the legs, completing the model and data.
        """
        self.xml = self._get_world_model_file()

        self.spec = mujoco.MjSpec.from_file(self.xml)

        # Fix the mesh and texture directory paths
        self.spec.compiler.meshdir = self.get_mesh_path()
        self.spec.compiler.texturedir = self.get_texture_path()

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
