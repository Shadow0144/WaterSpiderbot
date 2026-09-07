"""Class for managing neural network learning checkpoint files."""

import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory

import torch


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

    def get_model_weights_exists(self, filename):
        """Get if the model weight file exists."""
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, filename)
        return os.path.exists(full_filename)

    def save_actor_critic_weights(self,
                                  filename,
                                  actor,
                                  critic,
                                  actor_optimizer,
                                  critic_optimizer):
        """Save the learned weights to a file."""
        filepath = self.get_model_weights_path()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not filename.endswith('.pt'):
            filename = filename + '.pt'
        full_filename = os.path.join(filepath, filename)
        checkpoint = {
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'actor_optimizer_state_dict': actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, full_filename)

    def save_soft_actor_critics_weights(self,
                                        filename,
                                        actor,
                                        critic1,
                                        critic2,
                                        actor_optimizer,
                                        critic1_optimizer,
                                        critic2_optimizer):
        """Save the learned weights to a file."""
        filepath = self.get_model_weights_path()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not filename.endswith('.pt'):
            filename = filename + '.pt'
        full_filename = os.path.join(filepath, filename)
        checkpoint = {
            'actor_state_dict': actor.state_dict(),
            'critic1_state_dict': critic1.state_dict(),
            'critic2_state_dict': critic2.state_dict(),
            'actor_optimizer_state_dict': actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': critic2_optimizer.state_dict(),
        }
        torch.save(checkpoint, full_filename)

    def load_actor_critic_weights(self,
                                  filename,
                                  actor,
                                  critic,
                                  actor_optimizer,
                                  critic_optimizer,
                                  device):
        """Load the learned weights from a file."""
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
        if 'critic_state_dict' in checkpoint:
            critic.load_state_dict(
                checkpoint['critic_state_dict']
            )
        if 'actor_optimizer_state_dict' in checkpoint:
            actor_optimizer.load_state_dict(
                checkpoint['actor_optimizer_state_dict']
            )
        if 'critic_optimizer_state_dict' in checkpoint:
            critic_optimizer.load_state_dict(
                checkpoint['critic_optimizer_state_dict']
            )

    def load_soft_actor_critics_weights(self,
                                        filename,
                                        actor,
                                        critic1,
                                        critic2,
                                        actor_optimizer,
                                        critic1_optimizer,
                                        critic2_optimizer,
                                        device):
        """Load the learned weights from a file."""
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
        if 'critic1_state_dict' in checkpoint:
            critic1.load_state_dict(
                checkpoint['critic1_state_dict']
            )
        if 'critic2_state_dict' in checkpoint:
            critic2.load_state_dict(
                checkpoint['critic2_state_dict']
            )
        if 'actor_optimizer_state_dict' in checkpoint:
            actor_optimizer.load_state_dict(
                checkpoint['actor_optimizer_state_dict']
            )
        if 'critic1_optimizer_state_dict' in checkpoint:
            critic1_optimizer.load_state_dict(
                checkpoint['critic1_optimizer_state_dict']
            )
        if 'critic2_optimizer_state_dict' in checkpoint:
            critic2_optimizer.load_state_dict(
                checkpoint['critic2_optimizer_state_dict']
            )

    def reset_learned_actor_critic_weights(self):
        """Backup the current weights and start with new random weights."""
        time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.save_actor_critic_weights(f'test_weights_backup_{time_string}.pt')
        self.delete_saved_weights()

    def reset_learned_soft_actor_critics_weights(self):
        """Backup the current weights and start with new random weights."""
        time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.save_soft_actor_critics_weights(
            f'test_weights_backup_{time_string}.pt'
        )
        self.delete_saved_weights()

    def delete_saved_weights(self, filename):
        """Delete the saved weights file."""
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, filename)
        if os.path.exists(full_filename):
            os.remove(full_filename)

    def get_population_checkpoint_exists(self,
                                         filename):
        """Get if the model weight file exists."""
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, filename)
        return os.path.exists(full_filename)

    def save_population_checkpoint(self,
                                   filename,
                                   candidates,
                                   current_episode,
                                   parent_candidate_filename):
        """Save all the current candidate names and rewards."""
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, filename)
        with open(full_filename, 'w') as checkpoint_file:
            checkpoint_file.write(f'{current_episode}\n')
            if parent_candidate_filename is not None:
                checkpoint_file.write(parent_candidate_filename + '\n')
            else:
                checkpoint_file.write('\n')
            for candidate in candidates:
                filename = candidate.filename
                epoch_reward = candidate.epoch_reward
                checkpoint_file.write(f'{filename},{epoch_reward}\n')

    def load_population_checkpoint(self,
                                   filename):
        """Load the candidate names and rewards."""
        raw_candidates = []
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, filename)
        if (not os.path.exists(full_filename) or
           os.path.getsize(full_filename) == 0):
            raise FileNotFoundError('No model weights file found at '
                                    f'{full_filename}')
        with open(full_filename, 'r') as checkpoint_file:
            current_episode = int(checkpoint_file.readline())
            parent_candidate_filename = checkpoint_file.readline().strip()
            for row in checkpoint_file:
                items = row.split(',')
                raw_candidates.append([items[0], float(items[1])])
        return raw_candidates, current_episode, parent_candidate_filename

    def delete_population_checkpoint(self, filename):
        """Delete the saved population file."""
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, filename)
        if os.path.exists(full_filename):
            os.remove(full_filename)
