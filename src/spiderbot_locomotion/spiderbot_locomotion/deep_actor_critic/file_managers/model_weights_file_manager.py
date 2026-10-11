"""Class for managing neural network model weights files."""

import os

from ament_index_python.packages import get_package_share_directory

import torch


class ModelWeightsFileManager():
    """Class for managing neural network learning checkpoint files."""

    def __init__(self, filename, subfolder=''):
        """Store the subfolder."""
        self.filename = filename
        self.subfolder = subfolder
        if not self.filename.endswith('.pt'):
            self.filename = self.filename + '.pt'

    def get_model_weights_path(self):
        """Get the path to the model weights file from the share directory."""
        share_dir = get_package_share_directory('spiderbot_locomotion')
        model_path = os.path.join(share_dir, self.subfolder, 'model_weights')
        return model_path

    def get_model_weights_exists(self):
        """Get if the model weight file exists."""
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, self.filename)
        return os.path.exists(full_filename)

    def save_model_weights(
            self,
            actor,
            actor_optimizer,
            critics,
            critic_optimizers,
            filename_override=None
            ):
        """Save the learned weights to a file."""
        filename = (
            filename_override
            if filename_override else
            self.filename
        )
        filepath = self.get_model_weights_path()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        full_filename = os.path.join(filepath, filename)

        checkpoint = {
            'actor_state_dict': actor.state_dict(),
            'actor_optimizer_state_dict': actor_optimizer.state_dict(),
            'critic_state_dicts': [critic.state_dict() for critic in critics],
            'critic_optimizer_state_dicts': (
                [critic_optimizer.state_dict()
                 for critic_optimizer in critic_optimizers]
            ),
        }

        torch.save(checkpoint, full_filename)

    def load_model_weights(
            self,
            actor,
            actor_optimizer,
            critics,
            critic_optimizers,
            device,
            filename_override=None
            ):
        """Load the learned weights from a file."""
        filename = (
            filename_override
            if filename_override else
            self.filename
        )
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, filename)
        if not os.path.exists(full_filename):
            raise FileNotFoundError('No model weights file found at '
                                    f'{full_filename}')

        checkpoint = torch.load(full_filename, map_location=device)

        if 'actor_state_dict' in checkpoint:
            actor.load_state_dict(
                checkpoint['actor_state_dict']
            )
        if 'actor_optimizer_state_dict' in checkpoint:
            actor_optimizer.load_state_dict(
                checkpoint['actor_optimizer_state_dict']
            )
        if 'critic_state_dicts' in checkpoint:
            for critic, state_dict in zip(
                critics, checkpoint['critic_state_dicts']
            ):
                critic.load_state_dict(state_dict)
        if 'critic_optimizer_state_dicts' in checkpoint:
            for critic_optimizer, state_dict in zip(
                critic_optimizers, checkpoint['critic_optimizer_state_dicts']
            ):
                critic_optimizer.load_state_dict(state_dict)

    def delete_model_weights(self):
        """Delete the saved weights file."""
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, self.filename)
        if os.path.exists(full_filename):
            os.remove(full_filename)
