"""Set of converter functions to translate between ROS types and Python."""

from sensor_msgs.msg import JointState

from spiderbot_interfaces.msg import LegPose
from spiderbot_interfaces.msg import SpiderbotPose


def convert_vector3_to_list(vector3):
    """Convert geometry_msg Vector3 to a Python list."""
    return [vector3.x, vector3.y, vector3.z]


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
