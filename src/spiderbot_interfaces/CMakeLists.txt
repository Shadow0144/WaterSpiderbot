cmake_minimum_required(VERSION 3.28)

project(spiderbot_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/LegPose.msg"
    "msg/LegTargets.msg"
    "msg/SpiderbotPose.msg"
    "srv/GetSpiderbotDescription.srv"
    DEPENDENCIES
        geometry_msgs
        sensor_msgs
)

ament_package()