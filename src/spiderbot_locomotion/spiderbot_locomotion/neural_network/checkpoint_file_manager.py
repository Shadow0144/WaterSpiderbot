"""Class for managing neural network learning checkpoint files."""

import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory

import torch

from .deep_actor_critic import DeepActorCritic


class CheckpointFileManager():
    """Class for managing neural network learning checkpoint files."""

    def __init__(self):
        """Nothing to initialize."""
        pass

    def get_model_weights_path(self):
        """Get the path to the model weights file from the share directory."""
        share_dir = get_package_share_directory('spiderbot_locomotion')
        model_path = os.path.join(share_dir, 'model_weights')
        return model_path

    def get_model_weights_exist(self, filename='test_weights.pt'):
        """Get if the model weight file exists."""
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, filename)
        return os.path.exists(full_filename)

    def save_weights(self,
                     actor_critic,
                     optimizer,
                     filename='test_weights.pt'):
        """Save the learned weights to a file."""
        filepath = self.get_model_weights_path()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        full_filename = os.path.join(filepath, filename)
        checkpoint = {
            'actor_critic_state_dict': actor_critic.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, full_filename)

    def load_weights(self,
                     actor_critic,
                     optimizer,
                     device,
                     filename='test_weights.pt'):
        """Load the learned weights from a file."""
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, filename)
        if not os.path.exists(full_filename):
            raise FileNotFoundError('No model weights file found at '
                                    f'{full_filename}')

        checkpoint = torch.load(full_filename, map_location=device)

        if 'actor_critic_state_dict' in checkpoint:
            actor_critic.load_state_dict(
                checkpoint['actor_critic_state_dict']
            )
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(
                checkpoint['optimizer_state_dict']
            )

    def reset_learned_weights(self):
        """Backup the current weights and start with new random weights."""
        time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.save_weights(f'test_weights_backup_{time_string}.pt')
        self.delete_saved_weights()
        # Create a new critic and optimizer with random weights
        return (
            DeepActorCritic().to(self.device),
            torch.optim.Adam(
                self.actor_critic.parameters(),
                lr=1e-4)
        )

    def delete_saved_weights(self, filename='test_weights.pt'):
        """Delete the saved weights file."""
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, filename)
        if os.path.exists(full_filename):
            os.remove(full_filename)
