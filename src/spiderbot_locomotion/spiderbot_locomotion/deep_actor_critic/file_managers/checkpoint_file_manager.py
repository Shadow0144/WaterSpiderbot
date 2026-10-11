"""Class for managing neural network learning population checkpoint files."""

import os

from ament_index_python.packages import get_package_share_directory


class CheckpointFileManager():
    """Class for managing neural network learning checkpoint files."""

    def __init__(self, filename='checkpoint.csv', subfolder=''):
        """Store the subfolder."""
        self.filename = filename
        self.subfolder = subfolder
        if not self.filename.endswith('.csv'):
            self.filename = self.filename + '.csv'

    def get_population_checkpoint_path(self):
        """Get the path to the model weights file from the share directory."""
        share_dir = get_package_share_directory('spiderbot_locomotion')
        model_path = os.path.join(share_dir, self.subfolder, 'model_weights')
        return model_path

    def get_population_checkpoint_exists(self):
        """Get if the model weight file exists."""
        filepath = self.get_population_checkpoint_path()
        full_filename = os.path.join(filepath, self.filename)
        return os.path.exists(full_filename)

    def save_population_checkpoint(
            self,
            candidates,
            current_episode,
            current_candidate,
            current_generation
            ):
        """Save all the current candidate names and rewards."""
        filepath = self.get_population_checkpoint_path()
        full_filename = os.path.join(filepath, self.filename)
        with open(full_filename, 'w') as checkpoint_file:
            checkpoint_file.write(f'{current_episode}\n')
            checkpoint_file.write(f'{current_candidate}\n')
            checkpoint_file.write(f'{current_generation}\n')
            for candidate in candidates:
                filename = candidate.filename
                candidate_reward = candidate.candidate_reward
                checkpoint_file.write(f'{filename},{candidate_reward}\n')

    def load_population_checkpoint(self):
        """Load the candidate names and rewards."""
        candidate_filenames = []
        candidate_rewards = []
        filepath = self.get_population_checkpoint_path()
        full_filename = os.path.join(filepath, self.filename)
        if (not os.path.exists(full_filename) or
           os.path.getsize(full_filename) == 0):
            raise FileNotFoundError('No model weights file found at '
                                    f'{full_filename}')
        with open(full_filename, 'r') as checkpoint_file:
            current_episode = int(checkpoint_file.readline())
            current_candidate = int(checkpoint_file.readline())
            current_generation = int(checkpoint_file.readline())
            for row in checkpoint_file:
                items = row.split(',')
                candidate_filenames.append([items[0]])
                candidate_rewards.append(float(items[1]))
        return (
            candidate_filenames,
            candidate_rewards,
            current_episode,
            current_candidate,
            current_generation
        )

    def delete_population_checkpoint(self):
        """Delete the saved population file."""
        filepath = self.get_population_checkpoint_path()
        full_filename = os.path.join(filepath, self.filename)
        if os.path.exists(full_filename):
            os.remove(full_filename)
