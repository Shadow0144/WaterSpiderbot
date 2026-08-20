"""Set of converter functions to translate between ROS types and Python."""

from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3

import mujoco

from nav_msgs.msg import Odometry

from sensor_msgs.msg import JointState

from spiderbot_interfaces.msg import LegPose
from spiderbot_interfaces.msg import LegTargets
from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.msg import SpiderbotTargetPose

from .spider_leg import SpiderLeg


def convert_list_to_vector3(to_convert):
    """Convert a Python list to a geometry_msg Vector3."""
    vector3 = Vector3(
        x=to_convert[0],
        y=to_convert[1],
        z=to_convert[2]
    )
    return vector3


def convert_vector3_to_list(to_convert):
    """Convert a geometry_msg Vector3 to a Python list."""
    return [to_convert.x, to_convert.y, to_convert.z]


def create_joint_state(name, qpose):
    """Construct a JointStateObject."""
    joint_state = JointState()
    joint_state.name = name
    joint_state.position = qpose
    return joint_state


def create_odometry(body):
    """Construct an Odometry message."""
    msg = Odometry()
    msg.pose.pose.position = Point(
        x=float(body.xpos[0]),
        y=float(body.xpos[1]),
        z=float(body.xpos[2])
    )
    # MuJoCo w,x,y,z -> ROS2 x,y,z,w
    msg.pose.pose.orientation = Quaternion(
        x=float(body.xquat[1]),
        y=float(body.xquat[2]),
        z=float(body.xquat[3]),
        w=float(body.xquat[0])
    )
    msg.twist.twist.angular = Vector3(
        x=float(body.cvel[0]),
        y=float(body.cvel[1]),
        z=float(body.cvel[2])
    )
    msg.twist.twist.linear = Vector3(
        x=float(body.cvel[3]),
        y=float(body.cvel[4]),
        z=float(body.cvel[5])
    )
    return msg


def construct_leg_pose_msg(leg, leg_name):
    """Construct a LegPose message."""
    qposes = leg.get_qposes()
    qvels = leg.get_qvels()
    xyz = leg.get_claw_xyz()
    rpy = leg.get_claw_rpy()
    leg_pose = LegPose()
    leg_pose.leg_name = leg_name
    leg_pose.coxa_qpos = qposes[0]
    leg_pose.femur_qpos = qposes[1]
    leg_pose.tibia_qpos = qposes[2]
    leg_pose.coxa_qvel = qvels[0]
    leg_pose.femur_qvel = qvels[1]
    leg_pose.tibia_qvel = qvels[2]
    leg_pose.claw_x = xyz[0]
    leg_pose.claw_y = xyz[1]
    leg_pose.claw_z = xyz[2]
    leg_pose.claw_roll = rpy[0]
    leg_pose.claw_pitch = rpy[1]
    leg_pose.claw_yaw = rpy[2]
    return leg_pose


def construct_pose_msg(timestamp,
                       body,
                       leg_names,
                       legs):
    """Construct a SpiderbotPose message."""
    msg = SpiderbotPose()
    msg.timestamp = timestamp
    msg.body_odometry = create_odometry(
        body
    )
    leg_poses = []
    for leg_name in leg_names:
        leg_poses.append(construct_leg_pose_msg(
            legs[leg_name], leg_name
        ))
    msg.leg_poses = leg_poses
    return msg


def construct_target_pose_msg_from_legs(timestamp, leg_names, legs):
    """Construct a SpiderbotTargetPose message from legs."""
    msg = SpiderbotTargetPose()
    msg.timestamp = timestamp
    leg_poses = []
    leg_poses = []
    for leg_name in leg_names:
        leg_poses.append(construct_leg_pose_msg(
            legs[leg_name], leg_name
        ))
    msg.leg_poses = leg_poses
    return msg


def construct_target_pose_msg(timestamp, leg_names, target_qposes):
    """Construct a SpiderbotTargetPose message from target angles."""
    msg = SpiderbotTargetPose()
    msg.timestamp = timestamp
    leg_poses = []
    for leg_name in leg_names:
        qposes = target_qposes[leg_name]
        leg_pose = LegPose()
        # Only set the name and qposes
        leg_pose.leg_name = leg_name
        leg_pose.coxa_qpos = qposes[0]
        leg_pose.femur_qpos = qposes[1]
        leg_pose.tibia_qpos = qposes[2]
        leg_poses.append(leg_pose)
    msg.leg_poses = leg_poses
    return msg


def construct_leg_targets_msg(timestamp, leg_names, target_points):
    """Construct a LegTargets message."""
    msg = LegTargets()
    msg.timestamp = timestamp
    leg_targets = []
    for leg_name in leg_names:
        current_target = (
            target_points[leg_name]
        )
        leg_target = convert_list_to_vector3(current_target)
        leg_targets.append(leg_target)
    msg.leg_targets = leg_targets
    return msg


def convert_spiderbot_description_to_lists(spiderbot_description):
    """Convert a SpiderbotDescription message into Python lists."""
    leg_descriptions = (
        spiderbot_description.leg_descriptions
    )
    leg_names = [
        leg_description.leg_name
        for leg_description in leg_descriptions
    ]
    segment_lengths_per_leg = {}
    for i, leg_name in enumerate(leg_names):
        segment_lengths_per_leg[leg_name] = (
            leg_descriptions[i].segment_lengths
        )
    return (leg_descriptions, leg_names, segment_lengths_per_leg)


def convert_spiderbot_description_to_variables(spiderbot_description):
    """Convert a Spiderbot Description to various variables."""
    leg_descriptions, leg_names, segment_lengths_per_leg = (
        convert_spiderbot_description_to_lists(
            spiderbot_description
        )
    )

    spec = mujoco.MjSpec.from_string(
        spiderbot_description.spec_xml
    )
    model = spec.compile()
    data = mujoco.MjData(model)

    body = data.body('cephalothorax')
    legs = {}
    for leg_name in leg_names:
        legs[leg_name] = SpiderLeg(
            leg_name,
            segment_lengths_per_leg[leg_name],
            model,
            data
        )

    return (
        leg_descriptions,
        leg_names,
        segment_lengths_per_leg,
        spec,
        model,
        data,
        body,
        legs
    )
