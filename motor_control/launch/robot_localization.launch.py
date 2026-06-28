import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    # 1. Lấy đường dẫn tới file config EKF
    config_file_path = os.path.join(
        get_package_share_directory('motor_control'),
        'config',
        'ekf.yaml'
    )

    # 2. Lấy đường dẫn tới file launch 'bringup' của TurtleBot3
    # (Bạn có thể cần đổi 'turtlebot3_bringup' nếu tên package khác)
    tb3_bringup_launch_path = os.path.join(
        get_package_share_directory('turtlebot3_bringup'),
        'launch',
        'turtlebot3_bringup.launch.py' # Hoặc 'robot.launch.py' tùy phiên bản
    )

    # === THAY ĐỔI QUAN TRỌNG ===
    
    # 3. Khai báo node 'bringup' và TẮT TF CỦA NÓ
    start_tb3_bringup_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(tb3_bringup_launch_path),
        launch_arguments={'publish_tf': 'false'}.items() # <-- TẮT TF TẠI ĐÂY
    )
    # Lưu ý: Nếu dùng Gazebo, bạn cũng phải tìm cách tắt TF của Gazebo
    # ví dụ: `turtlebot3_world.launch.py` với `publish_robot_state:=false`
    
    # 4. Khai báo node odometry bánh xe của BẠN
    wheel_odometry_node = Node(
        package='my_robot_pkg',
        executable='tb3_odometry_node',
        name='tb3_wheel_odometry'
    )

    # 5. Khai báo node robot_localization (EKF)
    robot_localization_node = Node(
       package='robot_localization',
       executable='ekf_node',
       name='ekf_filter_node',
       output='screen',
       parameters=[config_file_path] # EKF sẽ publish TF
    )

    return LaunchDescription([
        start_tb3_bringup_cmd,    # Chạy bringup (đã tắt TF)
        wheel_odometry_node,      # Chạy node của bạn (không có TF)
        robot_localization_node   # Chạy EKF (sẽ là node duy nhất pub TF)
    ])
