# WaterSpiderbot

A project for learning MuJoCo and maybe bringing me water.

![Spiderbot walking with a handcrafted locomotion module](./assets/spiderbot_walking.webp)

## Prerequisites

* [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/index.html)
* Python Dependencies:

```bash
pip install -r requirements.txt
```

## Build

```bash
colcon build && source install/setup.bash
```
or
```bash
colcon build --symlink-install && source install/setup.bash
```

## Launch

- Simulation:
`ros2 launch spiderbot_bringup spiderbot_simulation.launch.xml`
- Real:
`ros2 launch spiderbot_bringup spiderbot_real.launch.xml`

## Packages

- Brain: Provides high-level goal setting such as setting targets for training or setting targets for the Spiderbot to travel to (e.g. travel to water unload station, travel to water load/recharge station)
- Bringup: Provides launch files for starting all the relevant spiderbot nodes quickly for a task
- Cartography: Provides SLAM functionality, i.e. localization and mapping
- Description: Provides a static description of the Spiderbot model, e.g. the body component shapes and relations, actuator descriptions, etc.
- Hardware: Provides an interface for interacting with the physical hardware of a real Spiderbot, providing the status and signals of the different hardware and pushing commands to the actuators etc.
- Interfaces: Defines the ROS2 interfaces (messages, services, etc.)
- Locomotion: Provides functionality to move or rotate the Spiderbot based on its target location and direction and its current pose
- Planning: Provides functionality for path-planning and obstacle-avoidance
- Simulation: Provides a simulation of the Spiderbot for testing the Spiderbot in a controlled environment
- Utility: Provides various shared utility functionality to the other packages

## Nodes

### `spiderbot_brain`
* `brain_node`
 * Sets training targets for the locomotion node
---
### `spiderbot_cartography`
* `cartography_node`
 * WIP
---
### `spiderbot_description`
* `description_node`
 * Supplies the Spiderbot description when requested
---
### `spiderbot_hardware`
* `hardware_node`
 * WIP
---
### `spiderbot_locomotion`
* `deep_actor_critic_locomotion_node`
 * Uses a deep-learning neural network and actor-critic reinforcement learning pattern to attempt to locomote the Spiderbot to a target location and orientation
* `handcrafted_angles_locomotion_node`
 * Uses a series of hand-crafted target angles for each actuator to make the Spiderbot move forward
* `handcrafted_points_locomotion_node`
* Uses a series of hand-crafted target points to move the claws to to make the Spiderbot move forward
* `simple_sin_locomotion_node`
* Uses two phase-shifted sin functions to move the legs in a simple pattern to make the Spiderbot move forward
---
### `spiderbot_planning`
* `planning_node`
 * WIP
---
### `spiderbot_simulation`
* `simulation_node`
 * Creates a MuJoCo simulation and viewer to watch and interact with a simulation of the Spiderbot