"""Utility functions for neural networks."""

import numpy as np

import torch


def construct_input_vector(target, spiderbot_pose_msg, device):
    """Construct a Pytorch tensor from the target and Spiderbot pose."""
    body_odometry = spiderbot_pose_msg.body_odometry
    body_pose = body_odometry.pose.pose.position
    body_orientation = body_odometry.pose.pose.orientation
    body_pose_vel = body_odometry.twist.twist.linear
    body_orientation_vel = body_odometry.twist.twist.angular

    dx = target[0] - body_pose.x
    dy = target[1] - body_pose.y
    distance = np.hypot(dx, dy)
    divisor = max(1.0, distance)
    dist_x = dx / divisor
    dist_y = dy / divisor
    height = body_pose.z

    qx = body_orientation.x
    qy = body_orientation.y
    qz = body_orientation.z
    qw = body_orientation.w
    roll = np.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    yaw = np.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    dtheta = yaw - target[2]
    dist_theta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi  # Clamp to [-pi, pi]

    data = [
        dist_x,
        dist_y,
        height,
        dist_theta,
        roll,
        pitch,
        yaw,
        body_pose_vel.x,
        body_pose_vel.y,
        body_pose_vel.z,
        body_orientation_vel.x,
        body_orientation_vel.y,
        body_orientation_vel.z,
    ]
    data.extend(
        [
            spiderbot_pose_msg.tendon_slide_action,
        ]
    )
    for leg_pose in spiderbot_pose_msg.leg_poses:
        data.extend(
            [
                leg_pose.coxa_qpos,
                leg_pose.femur_qpos,
                leg_pose.tibia_qpos,
                leg_pose.tarsus_qpos,
                leg_pose.coxa_qvel,
                leg_pose.femur_qvel,
                leg_pose.tibia_qvel,
                leg_pose.tarsus_qvel,
            ]
        )
    data = torch.tensor(data, dtype=torch.float32, device=device)
    data = data.unsqueeze(0)
    return data
