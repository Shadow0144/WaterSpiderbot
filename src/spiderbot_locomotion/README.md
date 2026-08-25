# spiderbot_locomotion

Provides target pose angles for the Spiderbot to help it locomote to target locations/orientations.

All of the locomotion nodes are mutually exclusive. The deep_actor_critic_locomotion_node is the recommended node to start for providing locomotion as the other nodes simply attempt to move the Spiderbot forward without regard for a target.

## Nodes

The locomotion nodes all share the following properties:

- Parameters:
 * training_mode_enabled -> bool
  If enabled, the actor-critic will operate in training mode and train the neural network; if disabled, the actor-critic will be in production mode and use what it perceives as best action each step and not perform any training

- Publishers:
 * spiderbot_target_pose -> SpiderbotTargetPose
  Publishes the set of target angles for the actuators of the Spiderbot
 * set_leg_targets -> LegTargets
  Publishes targets the leg claws at attempting to reach, used by the simulation to draw mocap bodies for visualization when leg targets are used

- Subscriptions:
 * spiderbot_pose -> SpiderbotPose
  Listens for the current pose of the Spiderbot
 * training_target -> TrainingTarget
  Listens for a new training target to start a new training episode

- Services:
 * set_training_mode_enabled -> SetBool
  Enables or disables training mode

- Clients:
 * get_spiderbot_description -> GetSpiderbotDescription
  Requests a description of the spiderbot; the node will block on this request
 * reset_simulation -> Empty
  If in training mode, resets the current training episode to prepare for the next episode

### deep_actor_critic_locomotion_node

Uses an actor-critic model to train a neural network to provide leg poses to guide the spiderbot toward a target
Training is done using population-based evolutionary selection
Learned model weights are stored in the install/spiderbot_locomotion/share/spiderbot_locomotion/model_weights folder

Inherits the locomotion node interfaces

- Publishers:
 * current_step_reward -> Float64
  Publishes the reward from the latest training step
 * episode_reward -> Float64
  Publishes the total reward from the latest training episode

- Services:
 * reset_learned_weights -> Trigger
  Resets the learned weights by deleting all the learned weight files and starting over

### simple_sin_locomotion_node

Uses a simple set of two phase-shifted sin functions to move the actuators of the legs through a set of target angles

Inherits the locomotion node interfaces

### handcrafted_angles_locomotion_node

Uses two phase-shifted set of four states for each leg to move the Spiderbot forward where each state sets a target set of actuator angles and it interpolates between the last phase and the current phase

Inherits the locomotion node interfaces

### handcrafted_points_locomotion_node

Uses two phase-shifted set of four states for each leg to move the Spiderbot forward where each state sets a target set of coordinates for the leg's claw to reach towards and it interpolates between the last phase and the current phase

Inherits the locomotion node interfaces