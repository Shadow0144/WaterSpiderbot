"""Convenience class to calculate the reward for a single step of RL."""

import math


class RewardCalculator():
    """Convenience class to calculate the reward for a single step of RL."""

    def __init__(self):
        """Initialize the reward calculator."""
        # Previous reward function state variables
        self.previous_distance = None
        self.previous_angular_distance = None

        # Hyperparameters for ranges
        self.nominal_z = 0.4
        self.nominal_z_range = 0.2
        self.distance_convergence = 0.05
        self.foot_off_ground_z = 0.05
        self.foot_max_z_above_body = 0.1
        self.min_qvel = 0.02
        self.max_qvel = 2.0

        # Hyperparameters for penalty strengths
        self.position_penalty = 100.0
        self.angle_penalty = 0.01
        self.tilt_penalty = 1.0
        self.height_penalty = 0.5
        self.angle_speed_penalty = 1.0
        self.feet_planted_penalty = 1.0
        self.feet_too_high_penalty = 1.0
        self.arrival_reward = 1000.0

        # Terminate early conditions
        self.max_tilt = 0.8
        self.min_height = 0.1
        self.time_to_reach_target_s = 0.0
        self.time_left_s = self.time_to_reach_target_s

        self.episode_reward = 0.0

    def set_time_to_reach_target(self, time_to_reach_target_s):
        """Set the time estimated to reach the target."""
        self.time_to_reach_target_s = time_to_reach_target_s
        self.time_left_s = self.time_to_reach_target_s

    def start_new_training_episode(self):
        """Reset the internal state variables for the episode."""
        self.previous_distance = None
        self.previous_angular_distance = None
        self.episode_reward = 0.0
        self.time_left_s = self.time_to_reach_target_s

    def compute_step_reward(self,
                            target,
                            spiderbot_pose,
                            delta_time):
        """Calculate per-step reward."""
        if target is None:
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
        target_theta = target[2]

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

        current_distance = math.hypot(target[0] - position.x,
                                      target[1] - position.y)
        if self.previous_distance is None:
            self.previous_distance = current_distance

        current_angular_distance = abs(math.atan2(
            math.sin(yaw - target_theta),
            math.cos(yaw - target_theta)
        ))
        if self.previous_angular_distance is None:
            self.previous_angular_distance = current_angular_distance

        legs_off_ground = 0
        total_actuation_outside_of_range = 0
        legs_above_body = 0
        for leg_pose in spiderbot_pose.leg_poses:
            leg_actuator_speeds = actuator_speeds[leg_pose.leg_name]
            for actuator_speed in leg_actuator_speeds:
                if abs(actuator_speed) < self.min_qvel:
                    total_actuation_outside_of_range += (
                        abs(self.min_qvel) - actuator_speed
                    )
                elif abs(actuator_speed) > self.max_qvel:
                    total_actuation_outside_of_range += (
                        abs(actuator_speed) - self.max_qvel
                    )

            legs_off_ground += (
                1 if leg_pose.claw_z > self.foot_off_ground_z else 0
            )

            legs_above_body += (
                1 if leg_pose.claw_z > (
                    position.z + self.foot_max_z_above_body
                ) else 0
            )
        too_many_legs_off_ground = max(0, legs_off_ground - 4)

        reward_progress = (
            -self.position_penalty *
            (current_distance - self.previous_distance)
        )
        self.previous_distance = current_distance

        reward_facing = (
                -self.angle_penalty *
                (current_angular_distance - self.previous_angular_distance)
        )
        self.previous_angular_distance = current_angular_distance

        reward_tilt = (
            -self.tilt_penalty * tilt
        )

        reward_height = (
            -self.height_penalty * z_distance
        )

        reward_angle_speed = (
            -self.angle_speed_penalty * total_actuation_outside_of_range
        )

        reward_feet_planted = (
            -self.feet_planted_penalty * too_many_legs_off_ground
        )

        reward_feet_too_high = (
            -self.feet_too_high_penalty * legs_above_body
        )

        total_reward = (
            reward_progress +
            reward_facing +
            reward_tilt +
            reward_height +
            reward_angle_speed +
            reward_feet_planted +
            reward_feet_too_high
        )

        done = False
        self.time_left_s -= delta_time
        if self.time_left_s < 0.0:
            done = True
        if (
            abs(roll) > self.max_tilt or
            abs(pitch) > self.max_tilt or
            position.z < self.min_height
        ):
            total_reward -= 50.0
            done = True
        elif current_distance < self.distance_convergence:
            total_reward += self.arrival_reward
            done = True

        self.episode_reward += total_reward

        return total_reward, done
