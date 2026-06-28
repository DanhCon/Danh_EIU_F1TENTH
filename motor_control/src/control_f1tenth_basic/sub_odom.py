#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import csv
import numpy as np
import os
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit_node")

        self.L = 0.33
        self.Ld = 1.0
        self.kq = 1.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer , self)
        self.car_frame = "ego_racecar/base_link"
        self.map_frame = "map"

        self.sub_odom = self.create_subscription(Odometry , "/odom", self.odom_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        self.waypoints = []
        
        csv_path = "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        self.load_waypoints(csv_path)

        self.get_logger().info("Pure Pursuit da khoi dong")

    def load_waypoints(self, filename):
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row: continue
                    
                    # 2. Xử lý file dùng dấu cách hoặc dấu phẩy
                    # Nếu dòng có chứa dấu phẩy, dùng csv reader chuẩn
                    # Nếu không, tự cắt bằng split()
                    line_data = row
                    if len(row) == 1 and isinstance(row[0], str):
                         line_data = row[0].split()
                    
                    # 3. Thử chuyển đổi sang số, nếu lỗi (do gặp chữ 'x') thì bỏ qua
                    try:
                        x = float(line_data[0])
                        y = float(line_data[1])
                        self.waypoints.append([x, y])
                    except ValueError:
                        # Đây là nơi nó bắt lỗi chữ 'x' và tự động bỏ qua dòng tiêu đề
                        continue
                        
            self.get_logger().info(f"Da tai thanh cong {len(self.waypoints)} diem waypoint.")
        except Exception as e:
            self.get_logger().error(f"LOI DOC FILE: {e}")
            
    def odom_callback(self, msg:Odometry):
        target_global = self.get_target_point(msg.pose.pose.position.x, msg.pose.pose.position.y)

        target_local = self.transform_waypoint(target_global)

        if target_local is not None:
            x_local = target_local[0]
            y_local = target_local[1]

            Ld_square = x_local**2 + y_local**2
            
            curvature = 2.0 * y_local / Ld_square
            steering_angle = math.atan(curvature * self.L) * self.kq

            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = steering_angle

            if abs(steering_angle) > 0.35:
                drive_msg.drive.speed = 2.0
            else:
                drive_msg.drive.speed = 3.0
            self.pub_drive.publish(drive_msg)
    def transform_waypoint(self, target_point):
        # Kiểm tra dữ liệu đầu vào
        if target_point is None:
            return None

        try:
            # SỬA LỖI 1: Dùng Time(seconds=0) để yêu cầu 'transform mới nhất' một cách rõ ràng
            # Thay vì rclpy.time.Time() có thể gây lỗi
            now = rclpy.time.Time(seconds=0)
            
            transform = self.tf_buffer.lookup_transform(
                self.car_frame, 
                self.map_frame,
                now, 
                timeout=rclpy.duration.Duration(seconds=1.0) # Tăng timeout lên 1 chút cho an toàn
            )
            
            t = transform.transform.translation
            q = transform.transform.rotation

            # SỬA LỖI 2: Đảm bảo Quaternion là float thuần túy
            R = self.quat_to_rot(float(q.w), float(q.x), float(q.y), float(q.z))

            # SỬA LỖI 3: Ép kiểu target_point thành float rõ ràng để numpy không bị điên
            p_x = float(target_point[0])
            p_y = float(target_point[1])
            
            point_global = np.array([p_x, p_y, 0.0])
            point_translation = np.array([float(t.x), float(t.y), float(t.z)])

            # Phép biến đổi Affine: P_local = R * P_global + T
            point_local = (R @ point_global) + point_translation
            return point_local

        except TransformException as ex:
            # Chỉ in lỗi mỗi 1 giây 1 lần để đỡ rác màn hình (Throttle log)
            self.get_logger().warn(f"TF Error: {ex}", throttle_duration_sec=1.0)
            return None
        except Exception as e:
            # Bắt các lỗi lạ khác (như lỗi numpy)
            self.get_logger().error(f"Loi tinh toan Matrix: {e}")
            return None

    def get_target_point(self, robot_x, robot_y):
        if not self.waypoints:
            return [0,0]
        min_dist = float("inf") # nay la gi 
        closest_index = 0 

        for i, point in enumerate(self.waypoints):
            d = self.dist([robot_x, robot_y], point)
            if d < min_dist:
                min_dist = d
                closest_index = i
        target_point = None
        total_points = len(self.waypoints)

        for k in range (total_points):
            curr_idx = (closest_index + k) % total_points
            point = self.waypoints[curr_idx]

            d = self.dist([robot_x, robot_y],point)

            if d >= self.Ld:
                target_point = point
                break
        if target_point is None:
            target_point = self.waypoints[closest_index]

        return target_point
    def dist(self, p1, p2):
        return math.sqrt((p1[0] -  p2[0])**2 +(p1[1] - p2[1])**2)
    def quat_to_rot(self, q0, q1, q2, q3):
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        rot_matrix = np.array([[r00, r01, r02],
                               [r10, r11, r12],
                               [r20, r21, r22]])   
        return rot_matrix
def main (args = None):
    rclpy.init(args= args)
    try:
        node = PurePursuit()
        rclpy.spin(node)
    except Exception as e:
        print(f"toang roi luom oi {e}")
    finally:
        if "node" in locals():
            node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()