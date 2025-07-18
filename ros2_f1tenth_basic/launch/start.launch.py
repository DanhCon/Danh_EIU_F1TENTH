import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    client_node = Node(
        package="ros_basic",
        executable="client.py",
        name="simple_service_client",
        output="screen",
        parameters=[
            os.path.join(get_package_share_directory("ros_basic"),"config","robot.yaml")]
        )
    service_node = Node(
        package="test_again",
        executable="service.py",
        name="simple_service_server",
        output="screen",
    )
    cau_1 = Node(
        package="test_again",
        executable="cau_1.py",
        name="simple_service_server",
        output="screen",
    )
    cau_2 = Node(
        package="test_again",
        executable="cau_2.py",
        name="simple_service_server",
        output="screen",
    )
    cau_3 = Node(
        package="test_again",
        executable="cau_3.py",
        name="simple_service_server",
        output="screen",
    )

    cau_6 = Node(
        package="test_again",
        executable="cau_6.py",
        name="simple_service_server",
        output="screen",
    )

    cau_7 = Node(
        package="test_again",
        executable="cau_7.py",
        name="simple_service_server",
        output="screen",
    )

    return LaunchDescription([
        client_node,
        service_node,
        cau_1,
        cau_2,
        cau_3,
        cau_6,
        cau_7,
        ])