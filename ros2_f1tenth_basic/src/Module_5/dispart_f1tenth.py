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
        self.fov_min = math.radians(-130.0)
        self.fov_max = math.radians(130.0)
        
        self.car_width = 0.2             # Tăng lên để bảo vệ sườn xe (Xe 0.3m + 0.05m lề)
        self.disparity_threshold = 0.05   # Nhạy hơn để phát hiện mép tường sớm hơn
        self.safe_dist = 0.4              # Chỉ đi vào chỗ nào mà phía trước trống ít nhất 80cm
        self.prev_angle = 0.0
        self.SMOOTH_ALPHA = 0.4   # 0 = không smooth, 1 = không thay đổi
        self.ANGLE_DEADZONE = math.radians(9.0)  # Dưới 3° thì coi như thẳng
        
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
        original = ranges.copy()
        filtered = ranges.copy()
        
        # Chỉ tạo bubble khi vật cản đủ gần (< 2.0m)
        # Vật xa chỉ cần disparity lớn hơn mới react
        CLOSE_DIST = 2.0
        FAR_THRESHOLD_MULTIPLIER = 3.0  # Vật xa cần disparity lớn hơn 3x
        
        for i in range(len(original) - 1):
            disparity = abs(original[i] - original[i + 1])
            near_dist = min(original[i], original[i + 1])
            
            # Ngưỡng disparity thích nghi theo khoảng cách
            if near_dist < CLOSE_DIST:
                threshold = self.disparity_threshold          # 0.10m khi gần
            else:
                threshold = self.disparity_threshold * FAR_THRESHOLD_MULTIPLIER  # 0.30m khi xa
            
            if disparity > threshold:
                if original[i] < original[i + 1]:
                    actual_near = max(original[i], 0.1)
                    bubble_angle = 2 * math.atan(self.car_width / (2 * actual_near))
                    bubble_rays = int(bubble_angle / angle_increment)
                    for j in range(max(0, i - bubble_rays), i + 1):
                        filtered[j] = 0.0
                else:
                    actual_near = max(original[i + 1], 0.1)
                    bubble_angle = 2 * math.atan(self.car_width / (2 * actual_near))
                    bubble_rays = int(bubble_angle / angle_increment)
                    for j in range(i + 1, min(len(filtered), i + 1 + bubble_rays + 1)):
                        filtered[j] = 0.0
        
        return filtered

    def find_max_gap(self, ranges):
        """ Đã sửa: clamp gap_end về len-1 """
        max_start, max_end, max_length = 0, 0, 0
        curr_start, curr_length = -1, 0
        n = len(ranges)

        for i in range(n):
            if ranges[i] > self.safe_dist:
                if curr_start == -1:
                    curr_start = i
                curr_length += 1
            else:
                if curr_length > max_length:
                    max_length = curr_length
                    max_start = curr_start
                    max_end = i - 1          # i-1 luôn < n → an toàn
                curr_start = -1
                curr_length = 0

        # Xử lý gap cuối vòng lặp
        if curr_length > max_length:
            max_start = curr_start
            max_end = n - 1                  # ✅ Đảm bảo không vượt quá index

        return max_start, max_end

    def find_best_point(self, start_idx, end_idx, ranges):
        """
        ✅ Fix: Tìm vùng xa nhất, trả về trung tâm vùng đó.
        Khi toàn bộ gap bằng nhau → trả về trung tâm gap luôn.
        """
        if start_idx >= end_idx:
            return (start_idx + end_idx) // 2

        sub = ranges[start_idx:end_idx + 1]
        max_dist = max(sub)
        threshold = max_dist * 0.95  # Dùng % thay vì trừ tuyệt đối → ổn định hơn

        # Tìm vùng liên tục xa nhất (không phải tất cả max_indices)
        best_start, best_end = start_idx, start_idx
        curr_start = -1
        curr_len = 0
        best_len = 0

        for i, v in enumerate(sub):
            if v >= threshold:
                if curr_start == -1:
                    curr_start = i
                curr_len += 1
            else:
                if curr_len > best_len:
                    best_len = curr_len
                    best_start = curr_start
                    best_end = curr_start + curr_len - 1
                curr_start = -1
                curr_len = 0

        if curr_len > best_len:
            best_start = curr_start
            best_end = curr_start + curr_len - 1

        # Trả về TRUNG TÂM của vùng xa nhất
        return start_idx + (best_start + best_end) // 2

    def lidar_callback(self, data: LaserScan):
        start_idx = int((self.fov_min - data.angle_min) / data.angle_increment)
        end_idx   = int((self.fov_max - data.angle_min) / data.angle_increment)
        end_idx   = min(end_idx, len(data.ranges) - 1)

        ranges = list(data.ranges[start_idx:end_idx])
        ranges = self.preprocess_lidar(ranges, 5.0)
        ranges = self.extend_disparities(ranges, data.angle_increment)
        gap_start, gap_end = self.find_max_gap(ranges)

        if gap_start == 0 and gap_end == 0:
            self.publish_drive(0.0, 0.0)
            return

        relative_best_idx = self.find_best_point(gap_start, gap_end, ranges)
        global_best_idx   = start_idx + relative_best_idx
        raw_angle = data.angle_min + global_best_idx * data.angle_increment

        # ✅ Fix Bug #2: Dead zone - bỏ qua góc nhỏ → không cua sớm vì nhiễu nhỏ
        if abs(raw_angle) < self.ANGLE_DEADZONE:
            raw_angle = 0.0

        # ✅ Fix Bug #2: Exponential moving average → không nhảy góc đột ngột
        smooth_angle = (1 - self.SMOOTH_ALPHA) * raw_angle + self.SMOOTH_ALPHA * self.prev_angle
        self.prev_angle = smooth_angle

        self.get_logger().info(
            f"gap: ({gap_start},{gap_end}), best: {relative_best_idx}, "
            f"raw: {math.degrees(raw_angle):.1f}°, smooth: {math.degrees(smooth_angle):.1f}°"
        )

        if abs(smooth_angle) > math.radians(20.0):
            speed = 1.0
        elif abs(smooth_angle) > math.radians(10.0):
            speed = 1.5
        else:
            speed = 2.5

        self.publish_drive(smooth_angle, speed)

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