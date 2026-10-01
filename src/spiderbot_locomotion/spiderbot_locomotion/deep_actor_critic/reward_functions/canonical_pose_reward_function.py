"""Calculates the reward for a single step of RL based on a canonical pose."""

import math


class CanonicalPoseRewardFunction():
    """Convenience class to calculate the reward for a single step of RL."""

    def __init__(self, logger):
        """Initialize the reward calculator."""
        self.logger = logger

        # Previous reward function state variables
        self.previous_x = None
        self.previous_y = None
        self.previous_distance = None
        self.previous_angular_distance = None
        self.target_reached = False
        self.episode_terminated = False

        # Convergence parameters
        self.distance_convergence = 0.05
        self.angle_convergence = 0.05

        # Hyperparameters for ranges
        self.target_speed = 0.02
        self.nominal_z = 0.4
        self.nominal_z_range = 0.2

        # Hyperparameters for penalty strengths
        self.position_progress_reward = 100.0
        self.position_anti_progress_penalty = -10.0
        self.angle_progress_reward = 100.0
        self.angle_anti_progress_penalty = -10.0
        self.tilt_penalty = -1.0
        self.height_penalty = -0.5
        self.pose_divergence_penalty = -0.01
        self.arrival_reward = 100.0
        self.terminated_early_penalty = -1_000.0
        self.reward_component_labels = [
            'Progress',
            'Facing',
            'Tilt',
            'Height',
            'Pose',
            'Arrival',
        ]

        # Terminate early conditions
        self.max_tilt = 0.8
        self.min_height = 0.0

    def set_target(self, target):
        """Set the training target."""
        self.target = target
        self.target_reached = False

    def start_new_training_episode(self):
        """Reset the internal state variables for the episode."""
        self.previous_x = None
        self.previous_y = None
        self.previous_distance = None
        self.previous_angular_distance = None
        self.target = None
        self.target_reached = False
        self.episode_terminated = False

    def compute_step_reward(self,
                            training_status,
                            spiderbot_pose,
                            delta_time):
        """Calculate per-step reward."""
        if (
            self.target is None or
            self.target_reached or
            self.episode_terminated
        ):
            reward_component_values = [0.0] * len(self.reward_component_labels)
            training_status.step_reward = 0.0
            training_status.target_reached = self.target_reached
            training_status.episode_terminated = self.episode_terminated
            training_status.reward_component_labels = (
                self.reward_component_labels
            )
            training_status.reward_component_values = (
                reward_component_values
            )
            return  # Return early

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

        tilt = (roll**2 + pitch**2)

        min_z = self.nominal_z - self.nominal_z_range
        max_z = self.nominal_z + self.nominal_z_range
        if position.z < min_z:
            z_distance = min_z - position.z
        elif position.z > max_z:
            z_distance = position.z - max_z
        else:
            z_distance = 0.0

        current_distance = math.hypot(self.target[0] - position.x,
                                      self.target[1] - position.y)
        if self.previous_distance is None:
            self.previous_distance = current_distance

        current_angular_distance = abs(math.atan2(
            math.sin(yaw - target_theta),
            math.cos(yaw - target_theta)
        ))
        if self.previous_angular_distance is None:
            self.previous_angular_distance = current_angular_distance

        if current_distance > self.previous_distance:
            reward_progress = (
                self.position_anti_progress_penalty *
                (current_distance - self.previous_distance)
            )
        else:
            reward_progress = (
                self.position_progress_reward *
                (self.previous_distance - current_distance)
            )
        self.previous_x = position.x
        self.previous_y = position.y
        self.previous_distance = current_distance

        if current_angular_distance > self.previous_angular_distance:
            reward_facing = (
                    self.angle_anti_progress_penalty *
                    (current_angular_distance - self.previous_angular_distance)
            )
        else:
            reward_facing = (
                    self.angle_progress_reward *
                    (self.previous_angular_distance - current_angular_distance)
            )
        self.previous_angular_distance = current_angular_distance

        reward_tilt = (
            self.tilt_penalty * tilt
        )

        reward_height = (
            self.height_penalty * z_distance
        )

        qpose_divergence = 0.0
        for leg_pose in spiderbot_pose.leg_poses:
            qpose_divergence += (
                abs(leg_pose.coxa_qpos) +
                abs(leg_pose.femur_qpos) +
                abs(leg_pose.tibia_qpos)
            )
        reward_pose = qpose_divergence * self.pose_divergence_penalty

        reward_arrival = 0.0
        if (
            current_distance < self.distance_convergence and
            current_angular_distance < self.angle_convergence
        ):
            reward_arrival = self.arrival_reward
            self.target_reached = True

        reward_terminated = 0.0
        self.episode_terminated = False
        if (
            abs(roll) > self.max_tilt or
            abs(pitch) > self.max_tilt or
            position.z < self.min_height
        ):
            reward_terminated = self.terminated_early_penalty
            self.episode_terminated = True

        total_reward = (
            reward_progress +
            reward_facing +
            reward_tilt +
            reward_height +
            reward_arrival +
            reward_pose +
            reward_terminated
        )

        reward_component_values = [
            reward_progress,
            reward_facing,
            reward_tilt,
            reward_height,
            reward_pose,
            reward_arrival,
        ]

        training_status.step_reward = total_reward
        training_status.target_reached = self.target_reached
        training_status.episode_terminated = self.episode_terminated
        training_status.reward_component_labels = self.reward_component_labels
        training_status.reward_component_values = reward_component_values
