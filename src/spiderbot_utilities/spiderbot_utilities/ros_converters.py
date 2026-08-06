"""Set of converter functions to translate between ROS types and Python."""

from geometry_msgs.msg import Vector3

from sensor_msgs.msg import JointState

from spiderbot_interfaces.msg import LegPose
from spiderbot_interfaces.msg import SpiderbotPose
from spiderbot_interfaces.msg import SpiderbotTargetPose


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


def construct_pose_msg(timestamp, bodyqpos, leg_names, legs):
    """Construct a pose message."""
    msg = SpiderbotPose()
    msg.timestamp = timestamp
    msg.body_joint_state = create_joint_state(
        'cephalothorax_joint',
        bodyqpos
    )
    leg_poses = []
    for leg_name in leg_names:
        qposes = legs[leg_name].get_qposes()
        leg_pose = LegPose()
        leg_pose.leg_name = leg_name
        leg_pose.coxa_qpos = qposes[0]
        leg_pose.femur_qpos = qposes[1]
        leg_pose.tibia_qpos = qposes[2]
        leg_poses.append(leg_pose)
    msg.leg_poses = leg_poses
    return msg


def construct_target_pose_msg_from_legs(timestamp, leg_names, legs):
    """Construct a pose message from legs."""
    msg = SpiderbotTargetPose()
    msg.timestamp = timestamp
    leg_poses = []
    for leg_name in leg_names:
        qposes = legs[leg_name].get_qposes()
        leg_pose = LegPose()
        leg_pose.leg_name = leg_name
        leg_pose.coxa_qpos = qposes[0]
        leg_pose.femur_qpos = qposes[1]
        leg_pose.tibia_qpos = qposes[2]
        leg_poses.append(leg_pose)
    msg.leg_poses = leg_poses
    return msg


def construct_target_pose_msg(timestamp, leg_names, target_qposes):
    """Construct a target pose message from target angles."""
    msg = SpiderbotTargetPose()
    msg.timestamp = timestamp
    leg_poses = []
    for leg_name in leg_names:
        qposes = target_qposes[leg_name]
        leg_pose = LegPose()
        leg_pose.leg_name = leg_name
        leg_pose.coxa_qpos = qposes[0]
        leg_pose.femur_qpos = qposes[1]
        leg_pose.tibia_qpos = qposes[2]
        leg_poses.append(leg_pose)
    msg.leg_poses = leg_poses
    return msg
