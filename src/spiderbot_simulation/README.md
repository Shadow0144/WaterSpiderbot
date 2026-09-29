# spiderbot_simulation

Provides a simulation and simulated view of a virtual Spiderbot using a MuJoCo viewer.

## Nodes

### simulation_node

Provides a simulation of the Spiderbot.

- Publishers:
 * spiderbot_pose -> SpiderbotPose
 Gives the current pose of the Spiderbot including actuator angles and the position and rotation of the main body

- Subscriptions:
 * set_leg_targets -> LegTargets
 Updates the visualization of the x,y,z targets for the end effector of the Spiderbot legs
 * spiderbot_target_pose -> SpiderbotTargetPose
 Sets the target actions for the actuators of the simulated Spiderbot
 * target -> Target
 Updates the visualization of where the Spiderbot's target (position and facing) is
 * target_number -> Int32
 Updates the overlays with information on the current target's number in the episode
 * training_status -> TrainingStatus
 Updates the overlays with information on the current status of the training

- Services:

 * reset_simulation -> Trigger
 Resets the Spiderbot back to its original pose by resetting the MuJoCo data

- Clients:
 * get_spiderbot_description -> GetSpiderbotDescription
  Requests a description of the spiderbot; the node will block on this request
  * get_training_configuration -> GetTrainingConfiguration
  Requests the training configuration for updating the overlays with training information if training is enabled