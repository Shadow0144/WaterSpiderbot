"""Utility functions for neural networks."""

import torch


def construct_input_vector(target, spiderbot_pose, device):
    """Construct a Pytorch tensor from the target and Spiderbot pose."""
    body_odometry = spiderbot_pose.body_odometry
    body_pose = body_odometry.pose.pose.position
    body_orientation = body_odometry.pose.pose.orientation
    body_pose_vel = body_odometry.twist.twist.linear
    body_orientation_vel = body_odometry.twist.twist.angular
    data = [
        target[0],  # x
        target[1],  # y
        target[2],  # theta
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
    data = torch.tensor(data, dtype=torch.float32, device=device)
    data = data.unsqueeze(0)
    return data
