#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.qos import qos_profile_sensor_data

class BasicDisparityExtender(Node):
    def __init__(self):
        super().__init__('basic_disparity_node')
        
        # --- THAM SỐ CƠ BẢN (Đã tinh chỉnh cho xe F1TENTH) ---
        self.fov_min = math.radians(-90.0)
        self.fov_max = math.radians(90.0)
        
        self.car_width = 0.40             # Bề rộng xe + lề an toàn (mét)
        self.disparity_threshold = 0.15   # Chênh lệch > 15cm thì coi là "Mép tường"
        self.safe_dist = 0.5              # Khoảng cách tối thiểu để coi là "Lối đi"
        
        # --- GIAO TIẾP ROS2 ---
        # Dùng qos_profile_sensor_data để khớp với môi trường Simulator
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        self.get_logger().info("🏎️ Basic Disparity Extender Đã Sẵn Sàng Chạy Mô Phỏng!")

    def preprocess_lidar(self, ranges, max_dist):
        """ Lọc nhiễu trung bình (Đã sửa lỗi kẹt index) """
        window_size = 5
        proc_ranges = [min(r, max_dist) if not math.isinf(r) and not math.isnan(r) else max_dist for r in ranges]        
        
        smooth_ranges = [0.0] * len(proc_ranges)
        for idx in range(len(proc_ranges)):
            # ĐÃ SỬA LỖI BUG CHÍ MẠNG Ở ĐÂY: Thay '1' bằng 'idx'
            start_idx = max(0, idx - window_size // 2)
            end_idx = min(len(proc_ranges), idx + window_size // 2 + 1)
            
            window_values = proc_ranges[start_idx:end_idx]
            smooth_ranges[idx] = sum(window_values) / len(window_values)
            
        return smooth_ranges

    def extend_disparities(self, ranges, angle_increment):
        """ Tìm mép tường và bơm phồng (Bubble) """
        filtered = ranges.copy()
        
        for i in range(len(filtered) - 1):
            # Tính khoảng chênh lệch giữa 2 tia kề nhau
            disparity = abs(filtered[i] - filtered[i + 1])
            
            if disparity > self.disparity_threshold:
                # Tìm ra mép tường (tia gần hơn)
                min_dist = min(filtered[i], filtered[i + 1])
                
                # Tính số lượng tia cần xóa (Bong bóng)
                bubble_angle = 2 * math.atan(self.car_width / (2 * min_dist))
                bubble_rays = int(bubble_angle / angle_increment)
                
                # Xóa các tia xung quanh mép tường (ép về 0.0)
                start_idx = max(0, i - bubble_rays // 2)
                end_idx = min(len(filtered) - 1, i + bubble_rays // 2)
                
                for j in range(start_idx, end_idx + 1):
                    filtered[j] = 0.0
                    
        return filtered

    def find_max_gap(self, ranges):
        """ Tìm khe hở rộng nhất """
        max_start, max_end, max_length = 0, 0, 0
        curr_start, curr_length = -1, 0
        
        for i in range(len(ranges)):
            if ranges[i] > self.safe_dist:
                if curr_start == -1: curr_start = i
                curr_length += 1
            else:
                if curr_length > max_length:
                    max_length = curr_length
                    max_start = curr_start
                    max_end = i - 1
                curr_start = -1
                curr_length = 0
                
        if curr_length > max_length:
            max_start = curr_start
            max_end = len(ranges) - 1
            
        return max_start, max_end

    def find_best_point(self, start_idx, end_idx, ranges):
        """ Chiến lược Naive: Chọn điểm xa nhất (sâu nhất) trong khe hở """
        best_idx = start_idx
        max_dist = 0.0
        
        for i in range(start_idx, end_idx + 1):
            if ranges[i] > max_dist:
                max_dist = ranges[i]
                best_idx = i
                
        return best_idx

    def lidar_callback(self, data: LaserScan):
        # 1. CẮT MẢNG THEO FOV (-90 đến +90)
        start_idx = int((self.fov_min - data.angle_min) / data.angle_increment)
        end_idx = int((self.fov_max - data.angle_min) / data.angle_increment)
        ranges = data.ranges[start_idx:end_idx]

        # 2. CHẠY PIPELINE
        ranges = self.preprocess_lidar(ranges, 5.0)
        ranges = self.extend_disparities(ranges, data.angle_increment)
        gap_start, gap_end = self.find_max_gap(ranges)
        
        # Nếu kẹt cứng không có lối thoát
        if gap_start == 0 and gap_end == 0:
            self.publish_drive(0.0, 0.0)
            return

        relative_best_idx = self.find_best_point(gap_start, gap_end, ranges)

        # 3. ĐÃ SỬA LỖI BUG CHÍ MẠNG: Bù lại vị trí index ban đầu đã bị cắt
        global_best_idx = start_idx + relative_best_idx
        angle = data.angle_min + global_best_idx * data.angle_increment
        
        # 4. CHIA TỐC ĐỘ THEO GÓC LÁI (Đơn giản hóa)
        if abs(angle) > math.radians(20.0):
            speed = 1.0 # Cua gắt
        elif abs(angle) > math.radians(10.0):
            speed = 2.5 # Cua nhẹ
        else:
            speed = 4.5 # Chạy thẳng
            
        # Gửi lệnh điều khiển
        self.publish_drive(angle, speed)

    def publish_drive(self, angle, speed):
        # ĐÃ SỬA LỖI: Cắt xén (Clamp) góc TRƯỚC KHI gán
        max_steer = math.radians(25.0)
        angle = max(-max_steer, min(max_steer, angle))

        msg = AckermannDriveStamped()
        
        # BẮT BUỘC TRONG SIMULATOR: Phải có header chứa thời gian thực
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        
        msg.drive.steering_angle = float(angle)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = BasicDisparityExtender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Dừng xe an toàn...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()