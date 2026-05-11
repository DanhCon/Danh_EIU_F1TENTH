#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import qos_profile_sensor_data

class BasicDisparityExtender(Node):
    def __init__(self):
        super().__init__('basic_disparity_node')
        
        # --- THAM SỐ CƠ BẢN ---
        self.fov_min = math.radians(-120.0)
        self.fov_max = math.radians(120.0)
        self.car_width = 0.6          
        self.disparity_threshold = 0.05   
        self.safe_dist = 0.4              
        self.prev_angle = 0.0
        self.SMOOTH_ALPHA = 0.2   
        self.ANGLE_DEADZONE = math.radians(2.0)

        # --- QUẢN LÝ VẬT CẢN ẢO ---
        self.virtual_obstacles = [] # Danh sách chứa {x, y, r}
        self.car_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        
        # --- GIAO TIẾP ROS2 ---
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        
        # Topic nhận điểm từ nút "Publish Point" trong RViz
        self.click_sub = self.create_subscription(PointStamped, '/clicked_point', self.click_callback, 10)
        # Topic dùng để xóa vật cản (Dùng nút "2D Nav Goal")
        self.clear_sub = self.create_subscription(PoseStamped, '/gqoal_pose', self.clear_callback, 10)
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/virtual_obstacles_markers', 10)
        
        self.create_timer(0.1, self.publish_markers)
        self.get_logger().info("🏎️ Interactive Disparity Extender Ready!")

    def odom_callback(self, msg: Odometry):
        """ Cập nhật vị trí xe để tính Raycasting """
        self.car_pose['x'] = msg.pose.pose.position.x
        self.car_pose['y'] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.car_pose['yaw'] = math.atan2(siny_cosp, cosy_cosp)

    def click_callback(self, msg: PointStamped):
        """ Thêm vật cản khi click chuột trên RViz """
        new_obs = {'x': msg.point.x, 'y': msg.point.y, 'r': 0.2}
        self.virtual_obstacles.append(new_obs)
        self.get_logger().info(f"📍 Thêm vật cản tại: {new_obs['x']:.2f}, {new_obs['y']:.2f}")

    def clear_callback(self, msg: PoseStamped):
        """ Xóa toàn bộ vật cản ảo """
        self.virtual_obstacles.clear()
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.marker_pub.publish(marker_array)
        self.get_logger().warn("🗑️ Đã xóa toàn bộ vật cản ảo")

    def inject_virtual_obstacles(self, ranges, angle_min, angle_increment):
        """ Bóp méo mảng LiDAR dựa trên vật cản ảo (Raycasting) """
        if not self.virtual_obstacles: return ranges
        
        new_ranges = np.array(ranges)
        angles = angle_min + np.arange(len(new_ranges)) * angle_increment
        
        for obs in self.virtual_obstacles:
            # Chuyển tọa độ vật cản về hệ tọa độ xe (Local)
            dx = obs['x'] - self.car_pose['x']
            dy = obs['y'] - self.car_pose['y']
            x_local = dx * math.cos(-self.car_pose['yaw']) - dy * math.sin(-self.car_pose['yaw'])
            y_local = dx * math.sin(-self.car_pose['yaw']) + dy * math.cos(-self.car_pose['yaw'])
            
            # Giải phương trình bậc 2 tìm giao điểm tia LiDAR và hình tròn
            dist_to_obs = math.hypot(x_local, y_local)
            if x_local < 0 or dist_to_obs > 10.0: continue
            
            # Tính góc bao phủ
            theta_center = math.atan2(y_local, x_local)
            delta_theta = math.asin(min(1.0, obs['r'] / dist_to_obs))
            
            mask = (angles > theta_center - delta_theta) & (angles < theta_center + delta_theta)
            indices = np.where(mask)[0]
            
            for i in indices:
                # Tính khoảng cách thực từ xe tới bề mặt vật cản ảo tại góc đó
                # Đơn giản hóa: dùng khoảng cách tới tâm trừ bán kính (hoặc giải pt bậc 2 chuẩn)
                virt_dist = dist_to_obs - obs['r']
                if virt_dist < new_ranges[i]:
                    new_ranges[i] = max(0.0, virt_dist)
                    
        return new_ranges.tolist()

    # --- CÁC HÀM XỬ LÝ LÕI (GIỮ NGUYÊN TỪ FILE CỦA BẠN) ---
    def preprocess_lidar(self, ranges, max_dist):
        window_size = 5
        proc_ranges = [min(r, max_dist) if not math.isinf(r) and not math.isnan(r) else max_dist for r in ranges]        
        smooth_ranges = [0.0] * len(proc_ranges)
        for idx in range(len(proc_ranges)):
            start_idx = max(0, idx - window_size // 2)
            end_idx = min(len(proc_ranges), idx + window_size // 2 + 1)
            window_values = proc_ranges[start_idx:end_idx]
            smooth_ranges[idx] = sum(window_values) / len(window_values)
        return smooth_ranges

    def extend_disparities(self, ranges, angle_increment):
        original = ranges.copy()
        filtered = ranges.copy()
        CLOSE_DIST, FAR_MULT = 2.0, 4.0
        for i in range(len(original) - 1):
            disparity = abs(original[i] - original[i + 1])
            near_dist = min(original[i], original[i + 1])
            threshold = self.disparity_threshold if near_dist < CLOSE_DIST else self.disparity_threshold * FAR_MULT
            if disparity > threshold:
                min_val = min(original[i], original[i + 1])
                bubble_angle = 2 * math.atan(self.car_width / (2 * max(min_val, 0.1)))
                bubble_rays = int(bubble_angle / angle_increment)
                if original[i] < original[i + 1]:
                    for j in range(max(0, i - bubble_rays), i + 1): filtered[j] = 0.0
                else:
                    for j in range(i + 1, min(len(filtered), i + 2 + bubble_rays)): filtered[j] = 0.0
        return filtered

    def find_max_gap(self, ranges):
        max_start, max_end, max_length = 0, 0, 0
        curr_start, curr_length = -1, 0
        for i, r in enumerate(ranges):
            if r > self.safe_dist:
                if curr_start == -1: curr_start = i
                curr_length += 1
            else:
                if curr_length > max_length:
                    max_length, max_start, max_end = curr_length, curr_start, i - 1
                curr_start, curr_length = -1, 0
        if curr_length > max_length:
            max_start, max_end = curr_start, len(ranges) - 1
        return max_start, max_end

    def find_best_point(self, start_idx, end_idx, ranges):
        if start_idx >= end_idx: return (start_idx + end_idx) // 2
        sub = ranges[start_idx:end_idx + 1]
        max_v = max(sub)
        thr = max_v * 0.4
        best_s, best_e, best_l, c_s, c_l = 0, 0, 0, -1, 0
        for i, v in enumerate(sub):
            if v >= thr:
                if c_s == -1: c_s = i
                c_l += 1
            else:
                if c_l > best_l:
                    best_l, best_s, best_e = c_l, c_s, c_s + c_l - 1
                c_s, c_l = -1, 0
        if c_l > best_l: best_s, best_e = c_s, c_s + c_l - 1
        return start_idx + (best_s + best_e) // 2

    def lidar_callback(self, data: LaserScan):
        s_idx = int((self.fov_min - data.angle_min) / data.angle_increment)
        e_idx = min(int((self.fov_max - data.angle_min) / data.angle_increment), len(data.ranges)-1)
        
        # TRƯỚC KHI XỬ LÝ: Tiêm vật cản ảo vào
        raw_ranges = list(data.ranges[s_idx:e_idx])
        injected_ranges = self.inject_virtual_obstacles(raw_ranges, self.fov_min, data.angle_increment)
        
        # Pipeline xử lý
        ranges = self.preprocess_lidar(injected_ranges, 5.0)
        ranges = self.extend_disparities(ranges, data.angle_increment)
        gap_s, gap_e = self.find_max_gap(ranges)
        
        if gap_s == 0 and gap_e == 0:
            self.publish_drive(0.0, 0.0); return

        best_idx = self.find_best_point(gap_s, gap_e, ranges)
        raw_angle = data.angle_min + (s_idx + best_idx) * data.angle_increment
        
        if abs(raw_angle) < self.ANGLE_DEADZONE: raw_angle = 0.0
        smooth_angle = (1 - self.SMOOTH_ALPHA) * raw_angle + self.SMOOTH_ALPHA * self.prev_angle
        self.prev_angle = smooth_angle

        # Tốc độ dựa trên góc lái
        speed = 4.0 if abs(smooth_angle) < math.radians(10.0) else (3.5 if abs(smooth_angle) < math.radians(20.0) else 2.0)
        self.publish_drive(smooth_angle, speed)

    def publish_drive(self, angle, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.steering_angle = float(max(-0.43, min(0.43, angle)))
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def publish_markers(self):
        """ Hiển thị vật cản ảo trên RViz """
        if not self.virtual_obstacles: return
        ma = MarkerArray()
        for i, obs in enumerate(self.virtual_obstacles):
            m = Marker()
            m.header.frame_id = "map"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = obs['x'], obs['y'], 0.2
            m.scale.x, m.scale.y, m.scale.z = obs['r']*2, obs['r']*2, 0.4
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 0.6
            ma.markers.append(m)
        self.marker_pub.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = BasicDisparityExtender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()