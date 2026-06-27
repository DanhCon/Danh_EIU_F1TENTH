#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import csv
import numpy as np
import os
# --- IMPORT MỚI CHO VISUALIZATION ---
from visualization_msgs.msg import Marker,MarkerArray
# ------------------------------------
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException
import tf2_geometry_msgs
from rclpy.duration import Duration
from geometry_msgs.msg import PointStamped
import tf2_ros
from copy import deepcopy


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit_node")

        # --- CÁC THAM SỐ QUAN TRỌNG CẦN CHỈNH ---
        self.L = 0.33    # [QUAN TRỌNG] Chiều dài trục cơ sở xe (F1Tenth ~0.33m). Đừng để 0!
        self.Ld = 1.0    # [QUAN TRỌNG] Khoảng cách nhìn trước (Lookahead). Thử 0.8 -> 1.2
        self.kq = 1.0    # Hệ số Gain góc lái.
        self.speed_fast = 7.0 # Tốc độ đường thẳng
        self.speed_medium = 4.5
        self.speed_slow = 4.5 # Tốc độ vào cuas
        # ----------------------------------------

        self.start_index = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer , self)
        self.car_frame = "ego_racecar/base_link"
        self.map_frame = "map"

        self.sub_odom = self.create_subscription(Odometry , "ego_racecar/odom", self.odom_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        
        # --- PUBLISHER MỚI CHO MARKER ---
        self.pub_marker_1 = self.create_publisher(Marker, "/lookahead_marker", 10)
        self.pub_marker_2 = self.create_publisher(Marker, "/publish_vi_tri_hien_tai", 10)
        self.pub_marker_3 = self.create_publisher(MarkerArray, "/publish_full_waypoint", 10)
        # --------------------------------

        self.waypoints = []
        # ĐƯỜNG DẪN FILE CSV CỦA BẠN (Giữ nguyên)
        csv_path = "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        self.load_waypoints(csv_path)
        self.publish_full_waypoint()

        self.get_logger().info(f"Pure Pursuit da khoi dong. L={self.L}, Ld={self.Ld}")

    # --- HÀM MỚI ĐỂ VẼ VISUALIZATION ---
    
    def find_giao_diem_voi_vong_tron_ahead(self,p1,p2,robot_pos,r):
        d = p2 -p1
        f = p1 - robot_pos

        a = np.dot(d,d)

        b = 2 * np.dot(f,d)
        c= np.dot(f,f) - r**2

        delta = b**2 -4*a*c

        if delta < 0:
            return None
        sqrt_dis = math.sqrt(delta)
        t1 = (-b - sqrt_dis) / (2*a)
        t2 = ( -b + sqrt_dis)/(2*a)

        if 0 <= t2 <= 1:
            return p1 + t2*d
        elif 0<= t1 <= 1:
            return p1 + t1*d
        return None
    def get_diem_lookahead(self, robot_x, robot_y):
        robot_pos = np.array([robot_x, robot_y])
        min_dist = float('inf')
        if not self.waypoints: 
            return None
        num_waypoints = len(self.waypoints)

        # 1. Tìm điểm gần nhất (Start Index)
        if self.start_index is None:
            # Lần đầu chạy: tìm toàn bộ
            min_dist = float("inf")
            for i, point in enumerate(self.waypoints):
                d = self.dist([robot_x, robot_y], point)
                if d < min_dist:
                    min_dist = d
                    self.start_index = i
        else:
            # Các lần sau: Chỉ tìm tiếp về phía trước (tránh nhảy cóc đường)
            curr_dist = self.dist([robot_x, robot_y], self.waypoints[self.start_index])
            for i in range (40):
                next_idx = (self.start_index + 1) % num_waypoints
                next_dist = self.dist([robot_x, robot_y], self.waypoints[next_idx])
                # Nếu điểm tiếp theo gần hơn điểm hiện tại, thì dịch chuyển tiếp
                if next_dist < curr_dist:
                    self.start_index = next_idx
                    curr_dist = next_dist
                else:
                    # Nếu điểm tiếp theo bắt đầu xa hơn, nghĩa là đã qua điểm gần nhất
                    break
        nearest_idx = self.start_index 
        search_window = 10
        lookahead_point = None
        for i in range(search_window):
        # Lấy chỉ số, cẩn thận vụ hết vòng lặp (index out of range)
            idx_start = (nearest_idx + i) % len(self.waypoints)
            idx_end = (nearest_idx + i + 1) % len(self.waypoints)
            
            p1 = np.array(self.waypoints[idx_start])
            p2 = np.array(self.waypoints[idx_end])
            
            # Gọi hàm toán học ở Bước 1
            intersection = self.find_giao_diem_voi_vong_tron_ahead(p1, p2, robot_pos, self.Ld)
            
            if intersection is not None:
                lookahead_point = intersection
                break # Tìm thấy giao điểm đầu tiên phía trước là dừng ngay!
                
        # 3. Fallback: Nếu không tìm thấy (do xe bị lệch quá xa đường đua)
        # Thì lấy đại một điểm waypoint ở xa làm lookahead (như cách cũ)
        # if lookahead_point is None:
        #     fallback_idx = (nearest_idx + 5) % len(self.waypoints)
        #     lookahead_point = np.array(self.waypoints[fallback_idx])
            
        return lookahead_point
        
    def publish_full_waypoint(self):
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
                    except ValueError:
                        continue
            self.waypoints = self.smooth_path(raw_waypoints, weight_data=0.5, weight_smooth=0.2)
            self.get_logger().info(f"Da tai thanh cong {len(self.waypoints)} diem waypoint.")

        except Exception as e:
            self.get_logger().error(f"LOI DOC FILE: {e}")
    def publish_lookahead_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame # Vẽ trên hệ tọa độ Map
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead_point"
        marker.id = 0
        marker.type = Marker.SPHERE  # Hình cầu
        marker.action = Marker.ADD
        
        # Kích thước quả cầu (30cm)
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        # Màu xanh lá (Green), đậm (alpha=1.0)
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1 # Nhấc lên khỏi mặt đất một chút
        marker.pose.orientation.w = 1.0

        self.pub_marker_1.publish(marker)
    # ------------------------------------
    def publish_vi_tri_hien_tai(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame # Vẽ trên hệ tọa độ Map
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "current_pos"
        marker.id = 0
        marker.type = Marker.SPHERE  # Hình cầu
        marker.action = Marker.ADD
        
        # Kích thước quả cầu (30cm)
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        # Màu xanh lá (Green), đậm (alpha=1.0)
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1 # Nhấc lên khỏi mặt đất một chút
        marker.pose.orientation.w = 1.0

        self.pub_marker_2.publish(marker)
            
    def odom_callback(self, msg: Odometry):
        # --- BƯỚC 1: LẤY VỊ TRÍ XE TRÊN HỆ TỌA ĐỘ MAP ---
        # Chúng ta KHÔNG dùng msg.pose.pose.position vì nó là hệ Odom
        try:
            # Hỏi TF: "Base_link đang ở đâu trên Map?"
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, 
                self.car_frame, 
                rclpy.time.Time(seconds=0), # Lấy mới nhất
                Duration(seconds=0.1) # Timeout ngắn thôi
            )
            
            # Đây mới là tọa độ thực của xe trên bản đồ
            robot_x_map = transform.transform.translation.x
            robot_y_map = transform.transform.translation.y

            self.publish_vi_tri_hien_tai(robot_x_map,robot_y_map)
            
        except TransformException as e:
            # Nếu chưa có TF map->base_link (ví dụ chưa 2D Pose Estimate), thì bỏ qua
            # self.get_logger().warn(f"Chua co vi tri tren Map: {e}", throttle_duration_sec=2.0)
            return
        # -----------------------------------------------

        # --- BƯỚC 2: TÌM ĐIỂM ĐÍCH (Dựa trên tọa độ Map vừa lấy) ---
        target_global = self.get_diem_lookahead(robot_x_map, robot_y_map)

        # --- VISUALIZATION ---
        if target_global is not None:
            self.publish_lookahead_marker(target_global[0], target_global[1])
        else:
            # Nếu không tìm thấy đích, dừng xe hoặc giữ nguyên lái
            return 
        # ---------------------

        # --- BƯỚC 3: CHUYỂN ĐỔI VỀ LOCAL (ĐỂ TÍNH GÓC LÁI) ---
        # Hàm transform_waypoint của bạn đã viết đúng, nó sẽ chuyển từ Map -> Base_link
        target_local = self.transform_waypoint(target_global)

        # --- BƯỚC 4: TÍNH TOÁN PURE PURSUIT ---
        if target_local is not None:
            x_local = target_local[0]
            y_local = target_local[1]

            Ld_square = x_local**2 + y_local**2
            
            # Tránh chia cho 0
            if Ld_square < 0.001: return

            curvature = 2.0 * y_local / Ld_square
            steering_angle = math.atan(curvature * self.L) * self.kq
            
            steering_angle = np.clip(steering_angle, -0.4, 0.4)

            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = steering_angle

            if  0.1 < abs(steering_angle) < 0.2:
                drive_msg.drive.speed = self.speed_slow
            elif 0.2 <= abs(steering_angle) < 0.3:
                drive_msg.drive.speed = self.speed_medium
            else:
                drive_msg.drive.speed = self.speed_fast
            
            self.pub_drive.publish(drive_msg)

    # --- HÀM TRANSFORM ĐÃ SỬA LỖI EXTRAPOLATION ---
    def transform_waypoint(self, target_point):
        if target_point is None: return None

        try:
            # Lấy Transform MỚI NHẤT (Time=0)
            transform = self.tf_buffer.lookup_transform(
                self.car_frame, 
                self.map_frame, 
                rclpy.time.Time(seconds=0), # Quan trọng: Lấy mới nhất
                Duration(seconds=1.0)
            )

            p_input = PointStamped()
            p_input.header.frame_id = self.map_frame
            # Quan trọng: Đồng bộ thời gian với transform vừa lấy
            p_input.header.stamp = transform.header.stamp 
            p_input.point.x = float(target_point[0])
            p_input.point.y = float(target_point[1])
            p_input.point.z = 0.0

            p_transformed = tf2_geometry_msgs.do_transform_point(p_input, transform)
            return np.array([p_transformed.point.x, p_transformed.point.y])

        except TransformException as e:
            # In lỗi ít hơn để đỡ rác màn hình
            self.get_logger().warn(f"TF Error (Transform): {e}", throttle_duration_sec=2.0)
            return None
        except Exception as e:
            self.get_logger().error(f"Lỗi lạ: {e}")
            return None
    # ---------------------------------------------
    # --------------------------------------------

    def dist(self, p1, p2):
        return math.sqrt((p1[0] -  p2[0])**2 +(p1[1] - p2[1])**2)
    # --- HÀM LÀM MƯỢT ĐƯỜNG (GRADIENT DESCENT) ---
    def smooth_path(self, path, weight_data=0.5, weight_smooth=0.2, tolerance=0.00001):
        """
        weight_data: Độ tin cậy vào dữ liệu gốc (càng lớn càng giống đường cũ)
        weight_smooth: Độ mượt (càng lớn càng thẳng)
        """
        # Tạo bản copy để không làm hỏng dữ liệu gốc ngay lập tức
        new_path = deepcopy(path)
        change = tolerance
        
        # Vòng lặp tối ưu hóa
        while change >= tolerance:
            change = 0.0
            # Không sửa điểm đầu (0) và điểm cuối (len-1) để giữ cố định đường đua
            for i in range(1, len(path) - 1):
                aux_x = new_path[i][0]
                aux_y = new_path[i][1]
                
                # Công thức Gradient Descent:
                # P_new = P_old + alpha*(Raw - P_old) + beta*(Neighbors - 2*P_old)
                
                new_path[i][0] += weight_data * (path[i][0] - new_path[i][0]) + \
                                  weight_smooth * (new_path[i-1][0] + new_path[i+1][0] - 2.0 * new_path[i][0])
                                  
                new_path[i][1] += weight_data * (path[i][1] - new_path[i][1]) + \
                                  weight_smooth * (new_path[i-1][1] + new_path[i+1][1] - 2.0 * new_path[i][1])
                
                change += abs(aux_x - new_path[i][0]) + abs(aux_y - new_path[i][1])
                
        return new_path

def main (args = None):
    rclpy.init(args= args)
    try:
        node = PurePursuit()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Lỗi không mong muốn: {e}")
    finally:
        if "node" in locals():
            # Gửi lệnh dừng xe trước khi thoát
            stop_msg = AckermannDriveStamped()
            node.pub_drive.publish(stop_msg)
            node.destroy_node()
        rclpy.shutdown()
        print("Pure Pursuit Node da tat.")

if __name__ == '__main__':
    main()