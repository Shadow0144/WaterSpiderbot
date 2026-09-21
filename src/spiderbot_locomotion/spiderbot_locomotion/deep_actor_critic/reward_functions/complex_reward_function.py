"""Convenience class to calculate the reward for a single step of RL."""

import math


class ComplexRewardFunction():
    """Convenience class to calculate the reward for a single step of RL."""

    def __init__(self, logger):
        """Initialize the reward calculator."""
        self.logger = logger

        # Previous reward function state variables
        self.previous_x = None
        self.previous_y = None
        self.previous_distance = None
        self.previous_angular_distance = None
        self.time_s = 0.0
        self.target_reached = False
        self.episode_terminated = False

        # Hyperparameters for ranges
        self.target_speed = 0.02
        self.nominal_z = 0.4
        self.nominal_z_range = 0.2
        self.distance_convergence = 0.05
        self.foot_off_ground_z = 0.05
        self.foot_max_z_above_body = 0.1
        self.min_qvel = 0.02
        self.max_qvel = 2.0

        # Hyperparameters for penalty strengths
        self.stationary_penalty = -100.0
        self.position_progress_reward = 1_000.0
        self.position_anti_progress_penalty = -100.0
        self.angle_progress_reward = 1_000.0
        self.angle_anti_progress_penalty = -0.1
        self.tilt_penalty = -1.0
        self.height_penalty = -0.5
        self.angle_speed_penalty = -0.01
        self.feet_raised_penalty = -0.01
        self.feet_too_high_penalty = -0.02
        self.terminated_early_penalty = -10.0
        self.arrival_reward = 30.0
        self.reward_component_labels = [
            'Progress',
            'Facing',
            'Movement',
            'Tilt',
            'Height',
            'Angle Speed',
            'Feet Planted',
            'Feet Too High',
        ]

        # Terminate early conditions
        self.max_tilt = 0.8
        self.min_height = 0.1

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
        self.time_s = 0.0
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
        self.time_s += delta_time

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

        actuator_speeds = {}
        for leg_pose in spiderbot_pose.leg_poses:
            actuator_speeds[leg_pose.leg_name] = [
                leg_pose.coxa_qvel,
                leg_pose.femur_qvel,
                leg_pose.tibia_qvel
            ]

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

        if self.previous_x is None:
            self.previous_x = position.x
        if self.previous_y is None:
            self.previous_y = position.y
        distance_traveled = math.hypot(self.previous_x - position.x,
                                       self.previous_y - position.y)
        if delta_time > 0.0:
            speed = distance_traveled / delta_time
        else:
            speed = 0.0
        target_speed_difference = (
            max(0.0, self.target_speed - speed)
        )

        legs_off_ground = 0
        actuators_not_actuating = 0
        legs_above_body = 0
        for leg_pose in spiderbot_pose.leg_poses:
            leg_actuator_speeds = actuator_speeds[leg_pose.leg_name]
            for actuator_speed in leg_actuator_speeds:
                if abs(actuator_speed) < self.min_qvel:
                    actuators_not_actuating += 1

            legs_off_ground += (
                1 if leg_pose.claw_z > self.foot_off_ground_z else 0
            )

            legs_above_body += (
                1 if leg_pose.claw_z > (
                    position.z + self.foot_max_z_above_body
                ) else 0
            )
        too_many_legs_off_ground = max(0, legs_off_ground - 4)

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

        reward_movement = (
                self.stationary_penalty *
                target_speed_difference
        )
        self.previous_x = position.x
        self.previous_y = position.y

        reward_tilt = (
            self.tilt_penalty * tilt
        )

        reward_height = (
            self.height_penalty * z_distance
        )

        reward_angle_speed = (
            self.angle_speed_penalty * actuators_not_actuating
        )

        reward_feet_planted = (
            self.feet_raised_penalty * too_many_legs_off_ground
        )

        reward_feet_too_high = (
            self.feet_too_high_penalty * legs_above_body
        )

        total_reward = (
            reward_progress +
            reward_facing +
            reward_movement +
            reward_tilt +
            reward_height +
            reward_angle_speed +
            reward_feet_planted +
            reward_feet_too_high
        )

        self.episode_terminated = False
        if (
            abs(roll) > self.max_tilt or
            abs(pitch) > self.max_tilt or
            position.z < self.min_height
        ):
            total_reward += self.terminated_early_penalty
            self.episode_terminated = True

        if current_distance < self.distance_convergence:
            total_reward += self.arrival_reward
            self.target_reached = True

        reward_component_values = [
            reward_progress,
            reward_facing,
            reward_movement,
            reward_tilt,
            reward_height,
            reward_angle_speed,
            reward_feet_planted,
            reward_feet_too_high,
        ]

        training_status.step_reward = total_reward
        training_status.target_reached = self.target_reached
        training_status.episode_terminated = self.episode_terminated
        training_status.reward_component_labels = self.reward_component_labels
        training_status.reward_component_values = reward_component_values
