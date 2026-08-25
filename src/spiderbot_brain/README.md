# spiderbot_brain

Provides high-level planning and goal setting for a Spiderbot. Currently, just generates training targets and times.

## Nodes

### brain_node

Provides the training targets and times.

- Parameters:
 * training_mode_enabled -> bool
  Sets if the Spiderbot is in training mode for training its locomotion or in production mode

- Publishers:
 * training_target -> TrainingTarget
  Publishes a target (x, y)-coordinate, a target facing (i.e. yaw), and a max time to reach that target

- Services:
 * set_training_mode_enabled -> SetBool
  Sets if the Spiderbot is in training mode for training its locomotion or in production mode