"""Locomotion neural network."""

import math
import os

from ament_index_python.packages import get_package_share_directory

import torch
from torch import nn

from .actor_critic import LocomotionActorCritic


class LocomotionNeuralNetwork():
    """Manages and executes a neural network for locomotion."""

    class IterationState:
        """Store training state information from a step."""

        def __init__(self,
                     action_np=None,
                     state_t_tensor=None,
                     probability_t_tensor=None,
                     value_t=None,
                     hidden_state_t=None,
                     hidden_state_tp1=None
                     ):
            """Store the values."""
            self.action_np = action_np
            self.state_t_tensor = state_t_tensor
            self.probability_t_tensor = probability_t_tensor
            self.value_t = value_t
            self.hidden_state_t = hidden_state_t
            self.hidden_state_tp1 = hidden_state_tp1

    def __init__(self):
        """Initialize the locomotion neural network."""
        self.device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available() else
            'cpu'
        )
        self.actor_critic = LocomotionActorCritic().to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(),
                                          lr=1e-4)

        # Recurrent hidden state
        self.hidden_state = None

        self.previous_distance = None
        self.previous_angular_distance = None

        self.nominal_z = 1.0  # TODO
        self.distance_convergence = 0.05
        self.foot_max_z = 0.1

        self.position_penalty = 1.0
        self.angle_penalty = 1.0
        self.tilt_penalty = 1.0
        self.height_penalty = -0.0
        self.feet_penalty = -1.0
        self.arrival_reward = 100.0

        self.gamma = 0.99

        self.time_to_goal_s = 0
        self.time_left_s = self.time_to_goal_s
        self.target = None

        self.latest_iter_state = None

    def reset(self):
        """Reset the internals if the training resets."""
        self.previous_distance = None
        self.previous_angular_distance = None
        self.hidden_state = None
        self.latest_iter_state = None

    def set_target(self, time_to_goal_s, target):
        """Update the target and the time expected to reach the goal."""
        self.time_to_goal_s = time_to_goal_s
        self.time_left_s = self.time_to_goal_s
        self.target = target

    def get_model_weights_exist(self, filename='test_weights.pt'):
        """Get if the file exists."""
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, filename)
        return os.path.exists(full_filename)

    def get_model_weights_path(self):
        """Get the path to the model weights file from the share directory."""
        share_dir = get_package_share_directory('spiderbot_locomotion')
        model_path = os.path.join(share_dir, 'model_weights')
        return model_path

    def save_weights(self, filename='test_weights.pt'):
        """Save the learned weights to a file."""
        filepath = self.get_model_weights_path()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        full_filename = os.path.join(filepath, filename)
        checkpoint = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, full_filename)

    def load_weights(self, filename='test_weights.pt'):
        """Load the learned weights from a file."""
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, filename)
        if not os.path.exists(full_filename):
            raise FileNotFoundError('No model weights file found at '
                                    f'{full_filename}')

        checkpoint = torch.load(full_filename, map_location=self.device)

        if 'actor_critic_state_dict' in checkpoint:
            self.actor_critic.load_state_dict(
                checkpoint['actor_critic_state_dict']
            )
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict']
            )

    def delete_saved_weights(self, filename='test_weights.pt'):
        """Delete the saved weights file."""
        filepath = self.get_model_weights_path()
        full_filename = os.path.join(filepath, filename)
        if os.path.exists(full_filename):
            os.remove(full_filename)

    def construct_input_vector(self, spiderbot_pose):
        """Construct a Pytorch tensor from the target and Spiderbot pose."""
        body_odometry = spiderbot_pose.body_odometry
        body_pose = body_odometry.pose.pose.position
        body_orientation = body_odometry.pose.pose.orientation
        body_pose_vel = body_odometry.twist.twist.linear
        body_orientation_vel = body_odometry.twist.twist.angular
        data = [
            self.target[0],  # x
            self.target[1],  # y
            self.target[2],  # theta
            body_pose.x,
            body_pose.y,
            body_pose.z,
            body_orientation.x,
            body_orientation.y,
            body_orientation.z,
            body_orientation.w,
            body_pose_vel.x,
            body_pose_vel.y,
            body_pose_vel.z,
            body_orientation_vel.x,
            body_orientation_vel.y,
            body_orientation_vel.z,
        ]
        for leg_pose in spiderbot_pose.leg_poses:
            data.extend(
                [
                    leg_pose.coxa_qpos,
                    leg_pose.femur_qpos,
                    leg_pose.tibia_qpos,
                    leg_pose.coxa_qvel,
                    leg_pose.femur_qvel,
                    leg_pose.tibia_qvel,
                    leg_pose.claw_x,
                    leg_pose.claw_y,
                    leg_pose.claw_z,
                    leg_pose.claw_roll,
                    leg_pose.claw_pitch,
                    leg_pose.claw_yaw,
                ]
            )
        data = torch.tensor(data, dtype=torch.float32, device=self.device)
        data = data.unsqueeze(0)
        return data

    def select_action(self, spiderbot_pose):
        """Step execution for training."""
        if self.target is None:
            return None  # Exit early if there is no target

        self.actor_critic.train()
        state_tensor = self.construct_input_vector(spiderbot_pose)

        action_dist, value_t, hidden_state_tp1 = self.actor_critic(
            state_tensor,
            self.hidden_state)

        action = action_dist.sample()
        log_probability = action_dist.log_prob(action).sum(dim=-1)

        action_np = action.squeeze(0).detach().cpu().numpy()

        hidden_state_t = self.hidden_state
        self.hidden_state = hidden_state_tp1.detach()

        return self.IterationState(
            action_np,
            state_tensor,
            log_probability,
            value_t,
            hidden_state_t,
            hidden_state_tp1
        )

    def forward(self, spiderbot_pose):
        """Short-hand function for a forward pass."""
        return self.select_action_deterministic(spiderbot_pose)

    def select_action_deterministic(self, spiderbot_pose):
        """Step execution for deployment."""
        self.actor_critic.eval()
        with torch.no_grad():
            state_tensor = self.construct_input_vector(spiderbot_pose)
            action_dist, _, hidden_state_tp1 = self.actor_critic(
                state_tensor, self.hidden_state
            )
            self.hidden_state = hidden_state_tp1
            return action_dist.mean.squeeze(0).cpu().numpy()

    def compute_step_reward(self,
                            spiderbot_pose,
                            delta_time):
        """Calculate per-step reward."""
        if self.target is None:
            return None, False  # Exit early if there is no target

        position = spiderbot_pose.body_odometry.pose.pose.position
        orientation = spiderbot_pose.body_odometry.pose.pose.orientation

        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        qw = orientation.w
        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        target_theta = self.target[2]

        current_distance = math.hypot(self.target[0] - position.x,
                                      self.target[1] - position.y)
        if self.previous_distance is None:
            self.previous_distance = current_distance

        current_angular_distance = math.atan2(math.sin(yaw - target_theta),
                                              math.cos(yaw - target_theta))
        if self.previous_angular_distance is None:
            self.previous_angular_distance = current_angular_distance

        legs_off_ground = 0
        for leg_pose in spiderbot_pose.leg_poses:
            legs_off_ground += 1 if leg_pose.claw_z > self.foot_max_z else 0
        legs_off_ground = max(0, legs_off_ground - 4)

        reward_progress = (
            self.position_penalty *
            (self.previous_distance - current_distance)
        )
        self.previous_distance = current_distance

        reward_facing = (
                self.angle_penalty *
                (self.previous_angular_distance - current_angular_distance)
        )
        self.previous_angular_distance = current_angular_distance

        reward_tilt = (
            -self.tilt_penalty * (roll**2 + pitch**2)
        )

        reward_height = (
            self.height_penalty * ((position.z - self.nominal_z)**2)
        )

        reward_feet_planted = (
            self.feet_penalty * legs_off_ground
        )

        total_reward = (
            reward_progress +
            reward_facing +
            reward_tilt +
            reward_height +
            reward_feet_planted
        )

        done = False
        self.time_left_s -= delta_time
        if self.time_left_s < 0:
            done = True
        if abs(roll) > 0.8 or abs(pitch) > 0.8 or position.z < 0.05:
            total_reward -= 50.0
            done = True
        elif current_distance < self.distance_convergence:
            total_reward += self.arrival_reward
            done = True

        return total_reward, done

    def train_step(self,
                   spiderbot_pose,
                   delta_time):
        """Perform a single step of training."""
        if self.target is None:
            return None, False  # Return early

        done = False
        if self.latest_iter_state is not None:
            # If there was a previous iter_state,
            # calculate the reward and train the actor-critic
            reward, done = self.compute_step_reward(
                    spiderbot_pose,
                    delta_time)

            next_data = self.construct_input_vector(
                spiderbot_pose)

            self.train_actor_critic_step(
                self.latest_iter_state,
                next_data,
                reward,
                done)

        self.latest_iter_state = (
            self.select_action(
                spiderbot_pose
            )
        )

        return self.latest_iter_state.action_np, done

    def train_actor_critic_step(self,
                                iter_state,
                                state_tp1_tensor,
                                reward,
                                done):
        """Perform a single-step Actor-Critic update."""
        self.actor_critic.train()

        reward_tensor = torch.tensor([reward],
                                     dtype=torch.float32,
                                     device=self.device)
        done_mask = torch.tensor([0.0 if done else 1.0],
                                 dtype=torch.float32,
                                 device=self.device)

        with torch.no_grad():
            hidden_state_for_next = (
                None if done else iter_state.hidden_state_tp1
            )
            _, value_tp1, _ = self.actor_critic(state_tp1_tensor,
                                                hidden_state_for_next)

        target_value = (
            reward_tensor + (self.gamma * value_tp1.squeeze(0) * done_mask)
        )

        advantage = target_value - iter_state.value_t.squeeze(0)

        actor_loss = -iter_state.probability_t_tensor * advantage.detach()

        critic_loss = self.loss_fn(iter_state.value_t.squeeze(0), target_value)

        total_loss = actor_loss + (0.5 * critic_loss)

        self.optimizer.zero_grad()
        total_loss.backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item()
