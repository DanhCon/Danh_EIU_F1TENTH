from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    move_robot_node = Node(
        package="ros_basic",
        executable="service.py",
        name="move_robot",
        output="screen",
        parameters=[
            {"quangduong": 4},
            {"khoangcach": 2.0}
        ],
    )

    return LaunchDescription([
        move_robot_node
    ])