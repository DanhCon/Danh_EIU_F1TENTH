#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import csv
import numpy as np
import os
import time
from copy import deepcopy

# --- IMPORT VISUALIZATION ---
from visualization_msgs.msg import Marker, MarkerArray
# ----------------------------

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs
from rclpy.duration import Duration
from geometry_msgs.msg import PointStamped

class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit_node")
        
        # --- CẤU HÌNH XE VÀ THUẬT TOÁN ---
        self.L = 0.39    # Chiều dài cơ sở (Wheelbase)
        self.kq = 1.0    # Hệ số Gain góc lái
        
        # --- CẤU HÌNH ADAPTIVE LOOKAHEAD (MỚI) ---
        # Công thức: Ld = clamp(speed * time, min, max)
        self.lookahead_time = 0.4    # T_lookahead (giây): Xe nhìn trước bao nhiêu giây?
        self.min_lookahead = 0.4     # l_min (mét): Khoảng cách tsối thiểu (để không lắc ở tốc độ thấp)
        self.max_lookahead = 4.0     # l_max (mét): Khoảng cách tối đa (để không bị mù ở tốc độ cao)
        self.Ld = self.min_lookahead # Khởi tạo giá trị mặc định



        self.last_curvature = 0.0
        self.Kd = 2.0
        # -----------------------------------------

        # Cấu hình vận tốc (Profile tuyến tính đơn giản)
        self.MAX_SPEED = 1.2
        self.MIN_SPEED = 1.2 # Bạn đang để min = max, xe sẽ chạy đều
        self.MAX_ANGLE = 0.35
        self.slope = (self.MIN_SPEED - self.MAX_SPEED) / self.MAX_ANGLE

        self.start_index = None

        # TF2 Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer , self)
        self.car_frame = "base_link"
        self.map_frame = "map"

        # Pub/Sub
        self.sub_odom = self.create_subscription(Odometry , "odom", self.odom_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.pub_text_marker = self.create_publisher(Marker, "/ten_xe", 10)
        
        # Markers
        self.pub_marker_1 = self.create_publisher(Marker, "/lookahead_marker", 10)
        self.pub_marker_2 = self.create_publisher(Marker, "/publish_vi_tri_hien_tai", 10)
        self.pub_marker_3 = self.create_publisher(MarkerArray, "/publish_duong_di", 10)

        self.waypoints = []
        # ĐƯỜNG DẪN FILE CSV (Giữ nguyên của bạn)
        csv_path = "/home/danh/ros2_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        
        # Kiểm tra file tồn tại để tránh crash
        if os.path.exists(csv_path):
            self.load_waypoints(csv_path)
            self.publish_duong_di()
        else:
            self.get_logger().error(f"Khong tim thay file CSV tai: {csv_path}")

        self.get_logger().info(f"Pure Pursuit Adaptive khoi dong. T_look={self.lookahead_time}s")

    def odom_callback(self, msg: Odometry):
        # --- BƯỚC 0: TÍNH TOÁN ADAPTIVE LOOKAHEAD (MỚI) ---
        # Lấy vận tốc hiện tại từ Odom
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        current_speed = math.hypot(vx, vy)

        # Tính Ld dựa trên công thức tuyến tính có bão hòa (Clamp)
        raw_ld = current_speed * self.lookahead_time
        self.Ld = np.clip(raw_ld, self.min_lookahead, self.max_lookahead)

        # Debug (có thể comment lại nếu spam log)
        # self.get_logger().info(f"Speed: {current_speed:.2f} m/s -> Ld: {self.Ld:.2f} m", throttle_duration_sec=1.0)
        # --------------------------------------------------

        # --- BƯỚC 1: LẤY VỊ TRÍ XE TRÊN HỆ TỌA ĐỘ MAP ---
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, 
                self.car_frame, 
                rclpy.time.Time(seconds=0),
                Duration(seconds=0.05)
            )
            robot_x_map = transform.transform.translation.x
            robot_y_map = transform.transform.translation.y

            self.publish_vi_tri_hien_tai(robot_x_map, robot_y_map)
            self.publish_ten_xe(robot_x_map, robot_y_map)
            
        except TransformException:
            return

        # --- BƯỚC 2: TÌM ĐIỂM ĐÍCH (Sử dụng self.Ld đã cập nhật ở Bước 0) ---
        target_global = self.get_diem_lookahead(robot_x_map, robot_y_map)

        if target_global is not None:
            self.publish_lookahead_marker(target_global[0], target_global[1])
        else:
            return 

        # --- BƯỚC 3: CHUYỂN ĐỔI VỀ LOCAL ---
        target_local = self.transform_waypoint(target_global)

        # --- BƯỚC 4: TÍNH TOÁN PURE PURSUIT ---
        if target_local is not None:
            x_local = target_local[0]
            y_local = target_local[1]
            Ld_square = x_local**2 + y_local**2
            if Ld_square < 0.001: return

            # --- TÍNH TOÁN PD ---
            # 1. Tính P (Curvature hiện tại)
            current_curvature = 2.0 * y_local / Ld_square

            # 2. Tính D (Tốc độ thay đổi curvature)
            # Lưu ý: Nếu xe đang đứng yên hoặc mới khởi động, change có thể vọt lên rất cao
            # nên có thể cần lọc nhiễu nếu servo bị rung.
            change_in_curvature = current_curvature - self.last_curvature
            
            # 3. Tổng hợp PD để ra độ cong mong muốn
            # Kp (self.kq) nhân với lỗi hiện tại
            # Kd (self.Kd) nhân với xu hướng thay đổi
            total_curvature = (current_curvature * self.kq) + (change_in_curvature * self.Kd)
            
            # 4. Lưu trạng thái cũ
            self.last_curvature = current_curvature
            
            # 5. Chuyển đổi sang góc lái (QUAN TRỌNG: Cần nhân với L)
            steering_angle = math.atan(total_curvature * self.L)
            
            # 6. Giới hạn góc lái vật lý
            steering_angle = np.clip(steering_angle, -0.35, 0.35)

            # --- GỬI LỆNH ---
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = steering_angle
            
            # Tính toán tốc độ (như cũ)
            abs_angle = abs(steering_angle)
            speed = self.slope * abs_angle + self.MAX_SPEED
            drive_msg.drive.speed = np.clip(speed, self.MIN_SPEED, self.MAX_SPEED)
        
            self.pub_drive.publish(drive_msg)

    # --- CÁC HÀM HỖ TRỢ (GIỮ NGUYÊN LOGIC CŨ) ---
    def find_giao_diem_voi_vong_tron_ahead(self, p1, p2, robot_pos, r):
        d = p2 - p1
        f = p1 - robot_pos
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - r**2
        delta = b**2 - 4*a*c
        if delta < 0: return None
        sqrt_dis = math.sqrt(delta)
        t1 = (-b - sqrt_dis) / (2*a)
        t2 = (-b + sqrt_dis) / (2*a)
        if 0 <= t2 <= 1: return p1 + t2*d
        elif 0 <= t1 <= 1: return p1 + t1*d
        return None

    def get_diem_lookahead(self, robot_x, robot_y):
        robot_pos = np.array([robot_x, robot_y])
        min_dist = float('inf')
        if not self.waypoints: return None
        num_waypoints = len(self.waypoints)

        if self.start_index is None:
            for i, point in enumerate(self.waypoints):
                d = self.dist([robot_x, robot_y], point)
                if d < min_dist:
                    min_dist = d
                    self.start_index = i
        else:
            curr_dist = self.dist([robot_x, robot_y], self.waypoints[self.start_index])
            for i in range (40):
                next_idx = (self.start_index + 1) % num_waypoints
                next_dist = self.dist([robot_x, robot_y], self.waypoints[next_idx])
                if next_dist < curr_dist:
                    self.start_index = next_idx
                    curr_dist = next_dist
                else:
                    break
                    
        nearest_idx = self.start_index 
        search_window = 40
        lookahead_point = None
        
        # Tìm giao điểm với vòng tròn bán kính self.Ld (Đã được cập nhật dynamic)
        for i in range(search_window):
            idx_start = (nearest_idx + i) % len(self.waypoints)
            idx_end = (nearest_idx + i + 1) % len(self.waypoints)
            p1 = np.array(self.waypoints[idx_start])
            p2 = np.array(self.waypoints[idx_end])
            
            intersection = self.find_giao_diem_voi_vong_tron_ahead(p1, p2, robot_pos, self.Ld)
            if intersection is not None:
                lookahead_point = intersection
                break
                
        if lookahead_point is None:
            fallback_idx = (nearest_idx + 5) % len(self.waypoints)
            lookahead_point = np.array(self.waypoints[fallback_idx])
            
        return lookahead_point

    def transform_waypoint(self, target_point):
        if target_point is None: return None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.car_frame, 
                self.map_frame, 
                rclpy.time.Time(seconds=0),
                Duration(seconds=1.0)
            )
            p_input = PointStamped()
            p_input.header.frame_id = self.map_frame
            p_input.header.stamp = transform.header.stamp 
            p_input.point.x = float(target_point[0])
            p_input.point.y = float(target_point[1])
            p_input.point.z = 0.0
            p_transformed = tf2_geometry_msgs.do_transform_point(p_input, transform)
            return np.array([p_transformed.point.x, p_transformed.point.y])
        except (TransformException, Exception) as e:
            # self.get_logger().warn(f"TF Error: {e}", throttle_duration_sec=2.0)
            return None

    def dist(self, p1, p2):
        return math.sqrt((p1[0] -  p2[0])**2 + (p1[1] - p2[1])**2)

    def load_waypoints(self, filename):
        raw_waypoints = []
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row: continue
                    line_data = row
                    if len(row) == 1 and isinstance(row[0], str):
                         line_data = row[0].split()
                    try:
                        x = float(line_data[0])
                        y = float(line_data[1])
                        raw_waypoints.append([x, y])
                    except ValueError: continue
            self.waypoints = self.smooth_path(raw_waypoints, weight_data=0.5, weight_smooth=0.5)
            self.get_logger().info(f"Da tai {len(self.waypoints)} diem waypoint.")
        except Exception as e:
            self.get_logger().error(f"LOI DOC FILE: {e}")

    def smooth_path(self, path, weight_data=0.5, weight_smooth=0.2, tolerance=0.00001):
        new_path = deepcopy(path)
        change = tolerance
        while change >= tolerance: 
            change = 0.0
            for i in range(1, len(path) - 1):
                aux_x = new_path[i][0]
                aux_y = new_path[i][1]
                new_path[i][0] += weight_data * (path[i][0] - new_path[i][0]) + \
                                  weight_smooth * (new_path[i-1][0] + new_path[i+1][0] - 2.0 * new_path[i][0])
                new_path[i][1] += weight_data * (path[i][1] - new_path[i][1]) + \
                                  weight_smooth * (new_path[i-1][1] + new_path[i+1][1] - 2.0 * new_path[i][1])
                change += abs(aux_x - new_path[i][0]) + abs(aux_y - new_path[i][1])
        return new_path

    # --- VISUALIZATION HELPERS ---
    def publish_ten_xe(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "text_info"
        marker.id = 999
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.scale.z = 1.0
        marker.color.a = 1.0; marker.color.r = 0.0; marker.color.g = 0.5; marker.color.b = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 2.8 # Cao hẳn lên
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.text = "EIU FABLAB"
        self.pub_text_marker.publish(marker)

    def publish_lookahead_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3
        marker.color.a = 1.0; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        self.pub_marker_1.publish(marker)

    def publish_vi_tri_hien_tai(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "current_pos"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3
        marker.color.a = 1.0; marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        self.pub_marker_2.publish(marker)

    def publish_duong_di(self):
        marker_array = MarkerArray()
        for i, point in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.1; marker.scale.y = 0.1; marker.scale.z = 0.1
            marker.color.a = 1.0; marker.color.r = 0.5; marker.color.g = 0.5
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker_array.markers.append(marker)
        self.pub_marker_3.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    except Exception as e: print(f"Error: {e}")
    finally:
        if rclpy.ok():
            try:
                stop_msg = AckermannDriveStamped()
                node.pub_drive.publish(stop_msg)
                time.sleep(0.1)
            except: pass
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()