# spiderbot_brain

Provides high-level planning and goal setting for a Spiderbot. Currently, just generates training targets and times.

## Nodes

### target_generator_node

Provides the training targets and times.

- Publishers:
 * target -> Target
  Publishes a target position (x, y) and facing (yaw) for the Spiderbot to locomote toward
 * target_number -> Int32
  Publishes the number of the current target in the episode
 * start_training_episode -> Empty
  Starts a new training episode in the locomotion node

- Subscriptions:
 * spiderbot_pose -> SpiderbotPose
  Provides the current pose of the Spiderbot
 * training_status -> TrainingStatus
  Provides information such as if a Spiderbot has reached their locomotion target or if its training episode has been terminated

- Services:
 * get_training_configuration -> GetTrainingConfiguration
  Provides the training configuration for the training session
 * enabled_training -> SetBool
  Enables (or disables) training when the locomotion node is ready to begin training

- Clients:
 * reset_simulation -> Trigger
  Resets the simulation between training episodes

### teleop_node

Provides an interface to teleoperate a Spiderbot.

- Publishers:
 * target -> Target
  Publishes a target position (x, y) and facing (yaw) for the Spiderbot to locomote toward

- Subscriptions:
 * spiderbot_pose -> SpiderbotPose
  Provides the current pose of the Spiderbot

- Services:
 * get_training_configuration -> GetTrainingConfiguration
  Provides a flag to disable training mode