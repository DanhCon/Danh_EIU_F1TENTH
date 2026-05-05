#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math

from sensor_msgs.msg import  LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

from visualization_msgs.msg import Marker , MarkerArray 
from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException
from rclpy.duration import Duration

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class FTG(Node):
    def __init__(self):
        super().__init__("FTG")

        self.fov_min = math.radians(-90)
        self.fov_max = math.radians(90)
        self.car_width = 0.4
        self.safe_dist = 0.5

        self.virtual_obstacles = []
        self.obs_id_conter = 0
        self.default_radius = 0.3

        self.car_x = 0.0
        self.car_y = 0.0
        self.car_yaW = 0.0

        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback)
        self.scan_sub = self.create_subscription(LaserScan,'/scan', self.lidar_callback)
        self.click_sub = self.create_subscription(PointStamped, '/clicked_point', self.click_callback)
        self.clear_obs_sub = self.create_subscription(PoseStamped, '/goal_pose', self.clear_obstacles_callback)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/virtual_obstacles', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.global_frame = 'map'

        self.robot_frame = 'base_link'

        self.create_timer(0.1, self.publish_markers)

        self.latest_ranges = None
        self.latest_scan_msg = None
        
        self.control_timer = self.create_timer(0.02, self.control_loop)
    def lidar_callback(self, msg:LaserScan):
        self.latest_scan_msg = msg 
        self.latest_ranges = np.array(msg.ranges)

    def click_callback(self, msg: PointStamped):
        """ 
        Bắt sự kiện 'Publish Point' từ RViz để thả vật cản ảo.
        msg chứa tọa độ (x, y, z) trên frame 'map'.
        """
        x, y = msg.point.x, msg.point.y
        
        # Tạo cấu trúc dữ liệu Dictionary cho vật cản
        new_obs = {
            'id': self.obs_id_counter, 
            'x': x, 
            'y': y, 
            'r': self.default_radius
        }
        
        self.virtual_obstacles.append(new_obs)
        self.obs_id_counter += 1 # Tăng ID để RViz phân biệt các object khác nhau
        
        # Log ra màn hình để debug
        self.get_logger().info(f"📍 DROPPED OBSTACLE [{new_obs['id']}] at X:{x:.2f}, Y:{y:.2f}")

    def clear_obstacles_callback(self, msg: PoseStamped):
        """ 
        Bắt sự kiện '2D Nav Goal' để reset môi trường.
        """
        count = len(self.virtual_obstacles)
        self.virtual_obstacles.clear() # Xóa data trong logic Node
        self.obs_id_counter = 0
        
        # --- Ép RViz dọn dẹp bộ nhớ đồ họa (VRAM) ---
        # Nếu chỉ xóa trong Node, hình ảnh 3D trên màn hình RViz vẫn còn (Ghost objects).
        # Ta phải gửi một lệnh DELETEALL đặc biệt.
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL # Lệnh hệ thống của RViz
        marker_array.markers.append(delete_marker)
        
        self.marker_pub.publish(marker_array)
        
        self.get_logger().info(f"🗑️ CLEARED {count} OBSTACLES")

    def update_robot_pose_from_tf(self) -> bool:
        try:
            t = self.tf_buffer.lookup_transform(
                target_frame= self.global_frame,
                source_frame= self.robot_frame,
                time= rclpy.time.Time(),
                timeout = Duration(seconds=0.05)
                

            )
            self.car_x = t.transform.translation.x
            self.car_y = t.transform.translation.y

            q = t.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)  # cho nay chua bt
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.car_yaw = math.atan2(siny_cosp, cosy_cosp)
            return True
        except TransformException as ex:
            self.get_logger().warn(
                f'TF Error: Không thể lấy transform từ {self.global_frame} sang {self.robot_frame}: {ex}',
                throttle_duration_sec=2.0
            )
            return False

    def control_loop(self):
        drive_msg = AckermannDriveStamped()

        if self.latest_ranges is None or self.latest_scan_msg is None:
            return
        if not self.update_robot_pose_from_tf():
            drive_msg.drive.steering_angle = 0.0
            drive_msg.drive.speed = 0.0
            self.drive_pub.publish(drive_msg)


            return 
        current_ranges = np.copy(self.latest_ranges)
        scan_msg = self.latest_scan_msg

        injected_ranges = self.inject_virtual_obstacles(current_ranges,scan_msg)
        steering_angel, speed = self.compute_ftg(injected_ranges, scan_msg)

        
        drive_msg.drive.steering_angle = float(steering_angel)
        drive_msg.drive.speed = float(speed)

        self.drive_pub.publish(drive_msg)
    def inject_virtual_obstacles(self, ranges:np.ndarray , scan_msg:LaserScan):
        if not self.virtual_obstacles:
            return ranges
        angles = scan_msg.angle_min + np.arrange(len(ranges)) * scan_msg.angle_increment

        for obs in self.virtual_obstacles:
            dx = obs['x'] - self.car_x
            dy = obs['y'] -self.car_y

            x_local = dx*math.cos(-self.car_yaw ) - dy*math.sin(-self.car_yaw)
            y_local = dx*math.sin(-self.car_yaw ) - dy*math.cos(-self.car_yaw)

            distance_to_obs = math.hypot(x_local, y_local)

            if x_local < 0 or distance_to_obs > 10.0:
                continue
            theta_center = math.atan2(y_local, x_local)

            
    

