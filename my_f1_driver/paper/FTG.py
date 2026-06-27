#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

# Import các Message cho UI Trigger[cite: 1, 2]
from geometry_msgs.msg import PointStamped, PoseStamped 

# Import QoS Profile để đồng bộ giao tiếp với RViz
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class InteractiveFGM(Node):
    def __init__(self):
        super().__init__('interactive_fgm_node')
        
        # --- 1. THAM SỐ THUẬT TOÁN FGM ---
        self.fov_min = math.radians(-90.0)
        self.fov_max = math.radians(90.0)
        self.car_width = 0.4
        self.safe_dist = 0.5
        
        # --- 2. QUẢN LÝ VẬT CẢN TƯƠNG TÁC ---
        self.virtual_obstacles = [] 
        self.obs_id_counter = 0     
        self.default_radius = 0.3   
        
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_yaw = 0.0
        
        # --- 3. CẤU HÌNH QoS (GIẢI QUYẾT LỖI MẤT GÓI TIN) ---
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # --- 4. ROS2 INTERFACES ---
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        
        # SUBSCRIBER 1: Thêm vật cản (Publish Point) - Áp dụng QoS
        self.click_sub = self.create_subscription(PointStamped, '/clicked_point', self.click_callback, qos_profile)
        
        # SUBSCRIBER 2: Xóa vật cản (2D Nav Goal)
        self.clear_obs_sub = self.create_subscription(PoseStamped, '/goal_pose', self.clear_obstacles_callback, 10)
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/virtual_obstacles', 10)
        
        self.create_timer(0.1, self.publish_markers)

    def click_callback(self, msg: PointStamped):
        """ Sinh vật cản ảo từ tín hiệu RViz """
        x, y = msg.point.x, msg.point.y
        new_obs = {'id': self.obs_id_counter, 'x': x, 'y': y, 'r': self.default_radius}
        self.virtual_obstacles.append(new_obs)
        self.obs_id_counter += 1
        self.get_logger().info(f"📍 DROPPED OBSTACLE [{new_obs['id']}] at X:{x:.2f}, Y:{y:.2f}")

    def clear_obstacles_callback(self, msg: PoseStamped):
        """ Kích hoạt khi dùng tool 2D Nav Goal trên RViz """
        count = len(self.virtual_obstacles)
        self.virtual_obstacles.clear()
        self.obs_id_counter = 0
        
        # Ép RViz dọn dẹp bộ nhớ đồ họa
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.marker_pub.publish(marker_array)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"🗑️ CLEARED {count} OBSTACLES")
        self.get_logger().info("=" * 60)

    def odom_callback(self, msg: Odometry):
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.car_yaw = math.atan2(siny_cosp, cosy_cosp)

    def inject_virtual_obstacles(self, msg: LaserScan) -> np.ndarray:
        """ Bóp méo mảng LiDAR thông qua Raycasting """
        ranges = np.array(msg.ranges)
        if not self.virtual_obstacles: return ranges 
            
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        cos_theta = np.cos(angles)
        sin_theta = np.sin(angles)
        
        for obs in self.virtual_obstacles:
            dx = obs['x'] - self.car_x
            dy = obs['y'] - self.car_y
            x_local = dx * math.cos(-self.car_yaw) - dy * math.sin(-self.car_yaw)
            y_local = dx * math.sin(-self.car_yaw) + dy * math.cos(-self.car_yaw)
            
            b = -2.0 * (x_local * cos_theta + y_local * sin_theta)
            c = x_local**2 + y_local**2 - obs['r']**2
            delta = b**2 - 4 * c
            valid_mask = delta >= 0
            
            if np.any(valid_mask):
                t1 = (-b[valid_mask] - np.sqrt(delta[valid_mask])) / 2.0
                update_mask = (t1 > 0.0) & (t1 < ranges[valid_mask])
                actual_indices = np.where(valid_mask)[0]
                ranges[actual_indices[update_mask]] = t1[update_mask]
        return ranges

    def preprocess_lidar(self, ranges, max_range):
        cleaned = np.clip(ranges, 0, max_range)
        window = np.ones(5) / 5.0
        return np.convolve(cleaned, window, mode='same').tolist()

    def create_safety_bubble(self, proc_ranges, angle_increment):
        min_dist = min(proc_ranges)
        min_idx = proc_ranges.index(min_dist)
        if min_dist < 0.05: min_dist = 0.05 
            
        bubble_angle = math.atan(self.car_width / min_dist)
        radius_idx = int(bubble_angle / angle_increment)
        start_idx = max(0, min_idx - radius_idx)
        end_idx = min(len(proc_ranges), min_idx + radius_idx + 1)
        for i in range(start_idx, end_idx): proc_ranges[i] = 0.0
        return proc_ranges

    def find_target_point(self, safe_ranges):
        max_gap_len, current_gap_len = 0, 0
        current_start, best_start, best_end = 0, 0, 0
        
        for i, r in enumerate(safe_ranges):
            if r > self.safe_dist:
                current_gap_len += 1
            else:
                if current_gap_len > max_gap_len:
                    max_gap_len = current_gap_len
                    best_start = current_start
                    best_end = i - 1
                current_gap_len = 0
                current_start = i + 1
                
        if current_gap_len > max_gap_len:
            best_start = current_start
            best_end = len(safe_ranges) - 1
            
        return (best_start + best_end) // 2

    def lidar_callback(self, msg: LaserScan):
        injected_ranges = self.inject_virtual_obstacles(msg)
        
        start_idx = int((self.fov_min - msg.angle_min) / msg.angle_increment)
        end_idx = int((self.fov_max - msg.angle_min) / msg.angle_increment)
        ranges_fov = injected_ranges[start_idx:end_idx]
        
        proc_ranges = self.preprocess_lidar(ranges_fov, msg.range_max)
        safe_ranges = self.create_safety_bubble(proc_ranges, msg.angle_increment)
        target_relative_idx = self.find_target_point(safe_ranges)
        
        global_target_idx = start_idx + target_relative_idx
        steering_angle = msg.angle_min + global_target_idx * msg.angle_increment
        
        max_steer = math.radians(25.0)
        steering_angle = max(-max_steer, min(max_steer, steering_angle))
        
        v_max, v_min = 5.0, 1.0
        steer_ratio = abs(steering_angle) / max_steer
        speed = v_max - (v_max - v_min) * (steer_ratio ** 1.5) 
            
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering_angle)
        drive_msg.drive.speed = float(speed)
        self.drive_pub.publish(drive_msg)

    def publish_markers(self):
        if not self.virtual_obstacles: return
            
        marker_array = MarkerArray()
        for obs in self.virtual_obstacles:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "virtual_obstacles"
            marker.id = obs['id']
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = obs['x']
            marker.pose.position.y = obs['y']
            marker.pose.position.z = 0.25
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = obs['r'] * 2.0 
            marker.scale.y = obs['r'] * 2.0
            marker.scale.z = 0.5            
            
            marker.color.r = 1.0 
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            marker_array.markers.append(marker)
            
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = InteractiveFGM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()