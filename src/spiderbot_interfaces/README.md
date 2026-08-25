# spiderbot_interfaces

Provides the ROS 2 message, service, and action interface definitions.

## Messages

- LegDescription
 Provides a name and a length for each segment of a leg, used by the Spiderbot description
- LegPose
 Provides a name, angle, and angular velocity for each of the leg actuators, as well as a position and orientation for the claw tip for a leg
- LegTargets
 Provides a timestamp and a set of coordinate for each leg that the leg is trying to position the claw at
- SpiderbotPose
 Provides a timestamp, a position and orientation of the Spiderbot body, and a pose for each of the legs
- SpiderbotTargetPose
 Provides a timestamp and a target pose for each of the legs
- TrainingTarget
 Provides a max time to reach a target pose, an x and y coordinate for the target, and an angle for the target

## Services

- GetSpiderbotDescription
 Asks the description node to provide a description of a Spiderbot, including the MuJoCo spec and a description of each leg