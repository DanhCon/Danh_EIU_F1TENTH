#!/usr/bin/env python3
import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description(): # bắt buộc
    client_go_lai_node = Node(
        package = "ros_basic", # tên package
        executable = "go_lai.py",
        name="distance",
        output="screen",
        parameters=[
            os.path.join(get_package_share_directory("ros_basic"),"config","robot.yaml")
        ]
    )
    service_go_lai_node = Node(
        package = "ros_basic",
        executable="go_lai_2.py",
        name = "SimpleServer",
        output = "screen",
        )
    
    return LaunchDescription([
        service_go_lai_node,
        client_go_lai_node
    ])