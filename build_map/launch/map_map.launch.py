import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
    # LƯU Ý: Đảm bảo bạn đã khai báo trong setup.py để copy folder 'maps' và 'config'
    # Nếu chưa làm setup.py, bạn có thể sửa thành đường dẫn tuyệt đối '/home/danh/...' để test nhanh
    pkg_share = get_package_share_directory('build_map')
    map_path = os.path.join(pkg_share, 'maps', 'map_fablab_new.yaml')
    amcl_config_path = os.path.join(pkg_share, 'config', 'amcl_config.yaml')
    
    return LaunchDescription([
        # --- 2. MAP SERVER (Hiện bản đồ) ---
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'use_sim_time': False}, 
                        {'yaml_filename': map_path}]
        ),

        # --- 3. AMCL (Định vị - Cho phép dùng 2D Pose Estimate) ---
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[
                amcl_config_path,   # Load tham số từ file config
                {'use_sim_time': False}
                # KHÔNG set initial_pose ở đây nữa để ưu tiên việc bấm tay
            ]
        ),

        # --- 4. LIFECYCLE MANAGER (Quản lý sống/chết của node) ---
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{'use_sim_time': False},
                        {'autostart': True},
                        {'node_names': ['map_server', 'amcl']}]
        ),

        # --- 5. RVIZ2 ---
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        ),
    ])