#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import csv
import numpy as np
import os
from copy import deepcopy

from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs
from rclpy.duration import Duration
from geometry_msgs.msg import PointStamped

class OpponentPurePursuit(Node):
    def __init__(self):
        super().__init__("opponent_pure_pursuit_node")
        
        # --- CẤU HÌNH XE ĐỐI THỦ (CHẠY CHẬM) ---
        self.L = 0.33    
        self.kq = 1.0    
        self.Ld = 1.0    # Cố định tầm nhìn lookahead cho đơn giản
        self.MAX_SPEED = 1.0  # TỐC ĐỘ CHẬM: 1.0 m/s (Để xe bạn có thể đuổi kịp)
        self.start_index = None

        # --- TF & TOPIC CỦA XE ĐỐI THỦ ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # LƯU Ý: Sử dụng frame và topic của opp_racecar
        self.car_frame = "opp_racecar/base_link" 
        self.map_frame = "map"
        
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "opp_drive", 10)
        self.timer = self.create_timer(0.05, self.timer_callback) # Chạy loop 20Hz

        # Tải Waypoint
        self.waypoints = []
        csv_path = "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        self.load_waypoints(csv_path)
        self.get_logger().info("Xe doi thu da san sang chay tren duong!")

    def timer_callback(self):
        try:
            # Lấy vị trí xe đối thủ
            transform = self.tf_buffer.lookup_transform(self.map_frame, self.car_frame, rclpy.time.Time())
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
        except TransformException:
            return

        target_global = self.get_diem_lookahead(robot_x, robot_y)
        if target_global is None: return

        # Transform waypoint về hệ trục xe đối thủ
        target_local = self.transform_waypoint(target_global)
        if target_local is not None:
            x_local, y_local = target_local[0], target_local[1]
            Ld_square = x_local**2 + y_local**2
            if Ld_square < 0.001: return

            # Tính toán góc lái Pure Pursuit
            curvature = 2.0 * y_local / Ld_square
            steering_angle = math.atan(curvature * self.L) * self.kq
            steering_angle = np.clip(steering_angle, -0.35, 0.35)

            # Đẩy lệnh điều khiển
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = steering_angle
            
            # Xe đối thủ rà phanh khi vào cua (giống xe bạn nhưng limit max nhỏ hơn)
            abs_angle = abs(steering_angle)
            speed = self.MAX_SPEED - (abs_angle * 1.5)
            drive_msg.drive.speed = np.clip(speed, 0.5, self.MAX_SPEED)
            
            self.pub_drive.publish(drive_msg)

    # --- CÁC HÀM XỬ LÝ TOÁN HỌC (GIỮ NGUYÊN BẢN GỐC) ---
    def transform_waypoint(self, target_point):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.car_frame, self.map_frame, rclpy.time.Time(), Duration(seconds=0.1)
            )
            p_input = PointStamped()
            p_input.header.frame_id = self.map_frame
            p_input.point.x, p_input.point.y = float(target_point[0]), float(target_point[1])
            p_transformed = tf2_geometry_msgs.do_transform_point(p_input, transform)
            return np.array([p_transformed.point.x, p_transformed.point.y])
        except Exception: return None

    def find_giao_diem(self, p1, p2, robot_pos, r):
        d, f = p2 - p1, p1 - robot_pos
        a, b, c = np.dot(d, d), 2 * np.dot(f, d), np.dot(f, f) - r**2
        delta = b**2 - 4*a*c
        if delta < 0: return None
        sqrt_dis = math.sqrt(delta)
        t1, t2 = (-b - sqrt_dis) / (2*a), (-b + sqrt_dis) / (2*a)
        if 0 <= t2 <= 1: return p1 + t2*d
        elif 0 <= t1 <= 1: return p1 + t1*d
        return None

    def dist(self, p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def get_diem_lookahead(self, robot_x, robot_y):
        robot_pos = np.array([robot_x, robot_y])
        if not self.waypoints: return None
        
        if self.start_index is None:
            min_dist = float('inf')
            for i, pt in enumerate(self.waypoints):
                d = self.dist(robot_pos, pt)
                if d < min_dist: min_dist, self.start_index = d, i
        else:
            curr_dist = self.dist(robot_pos, self.waypoints[self.start_index])
            for _ in range(40):
                next_idx = (self.start_index + 1) % len(self.waypoints)
                next_dist = self.dist(robot_pos, self.waypoints[next_idx])
                if next_dist < curr_dist:
                    self.start_index, curr_dist = next_idx, next_dist
                else: break

        for i in range(40):
            idx_start = (self.start_index + i) % len(self.waypoints)
            idx_end = (self.start_index + i + 1) % len(self.waypoints)
            intersection = self.find_giao_diem(
                np.array(self.waypoints[idx_start]), np.array(self.waypoints[idx_end]), robot_pos, self.Ld
            )
            if intersection is not None: return intersection
            
        return np.array(self.waypoints[(self.start_index + 5) % len(self.waypoints)])

    def load_waypoints(self, filename):
        raw_waypoints = []
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row: continue
                    line_data = row[0].split() if len(row) == 1 and isinstance(row[0], str) else row
                    try: raw_waypoints.append([float(line_data[0]), float(line_data[1])])
                    except ValueError: continue
            
            # Smooth nhanh
            self.waypoints = deepcopy(raw_waypoints)
            change = 1.0
            while change >= 0.0001: 
                change = 0.0
                for i in range(1, len(raw_waypoints) - 1):
                    aux_x, aux_y = self.waypoints[i][0], self.waypoints[i][1]
                    self.waypoints[i][0] += 0.5 * (raw_waypoints[i][0] - self.waypoints[i][0]) + 0.2 * (self.waypoints[i-1][0] + self.waypoints[i+1][0] - 2.0 * self.waypoints[i][0])
                    self.waypoints[i][1] += 0.5 * (raw_waypoints[i][1] - self.waypoints[i][1]) + 0.2 * (self.waypoints[i-1][1] + self.waypoints[i+1][1] - 2.0 * self.waypoints[i][1])
                    change += abs(aux_x - self.waypoints[i][0]) + abs(aux_y - self.waypoints[i][1])
        except Exception as e:
            self.get_logger().error(f"LOI DOC FILE: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = OpponentPurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()