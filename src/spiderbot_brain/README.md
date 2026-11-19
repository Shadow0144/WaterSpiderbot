# spiderbot_brain

Provides high-level planning and goal setting for a Spiderbot. Currently, just generates training targets and times.

## Nodes

### target_generator_node

Provides training information and targets for training the Spiderbot locomotion module.

- Parameters:
 * stationary_target -> bool
  If True, each target will generate with a position and rotation offset of [0.0, 0.0, 0.0];
  if False, each target will generate with an offset from the previous target plus some jitter,
  the offset changes every episode
 * relative_target -> bool
 If True, each target will generate relative to the Spiderbot's current position and rotation;
 if False, each target will generate relative to world origin

- Publishers:
 * locomotion_target -> LocomotionTarget
  Publishes a target position (x, y) and facing (yaw) for the Spiderbot to locomote toward
 * locomotion_target_number -> Int32
  Publishes the number of the current locomotion target in the episode
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
 * start_training -> Trigger
  Fired when the locomotion node is ready to begin training and starts the training process

### teleop_node

Provides an interface to teleoperate a Spiderbot.

- Publishers:
 * locomotion_target -> LocomotionTarget
  Publishes a target position (x, y) and facing (yaw) for the Spiderbot to locomote toward

- Subscriptions:
 * spiderbot_pose -> SpiderbotPose
  Provides the current pose of the Spiderbot

- Services:
 * get_training_configuration -> GetTrainingConfiguration
  Provides a flag to disable training mode