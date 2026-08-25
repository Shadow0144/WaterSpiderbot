# spiderbot_description

Provides static description of a Spiderbot including shapes, connectivity, actuator parameters, etc.

Models can be found in the models/ directory.

### Models

### spiderbot_base.xml

Provides the base for the Spiderbot description including the main body parts and eyes; the legs are added dynamically by the description_node.

## Nodes

### description_node

Provides the Spiderbot description.

- Services:
 * get_spiderbot_description -> GetSpiderbotDescription
  Provides a description of the Spiderbot, including the MuJoCo spec and a description of each leg.