
import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    
    Client = Node(
        package="motor_control",
        executable="service_client.py",
        name="async_client",
        output="screen",
        parameters=[
            os.path.join(get_package_share_directory("motor_control"),"config","robot.yaml")]
    )


    return LaunchDescription([

        Client, 

    ])

