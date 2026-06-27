#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import csv
import numpy as np
from copy import deepcopy

from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, Point
def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

def normalize_angle(angle):
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle

class MPCNode(Node):
    def __init__(self):
        super().__init__("mpc_controller_node")

        # ========================================================
        # 1. CÁC THAM SỐ VẬT LÝ VÀ MPC CHO F1TENTH
        # ========================================================
        self.m = 3.47       
        self.Iz = 0.04712   
        self.Caf = 60.0     
        self.Car = 60.0     
        self.lf = 0.158     
        self.lr = 0.171     
        self.Ts = 0.05      # Lấy mẫu 20Hz
        
        self.hz = 13        # Nhìn trước 1 giây (Đủ xa để biết cua gắt, đủ nhẹ cho CPU)
        self.outputs = 2    

        # 2. Xây "Bức tường ảo" ép xe bám vạch kẻ đường
        self.Q = np.array([[100.0, 0.0], 
                           [0.0, 100.0]])  # Q_y = 500: Phạt CỰC NẶNG nếu xe dám đi lệch vạch (chống cắt cua)
        self.S = np.array([[100.0, 0.0], 
                           [0.0, 100.0]])  
        
        # 3. Cấp "Vô lăng trợ lực siêu nhẹ" cho thuật toán
        self.R = np.array([[100.0]])         # Giảm R từ 100 xuống 5. Cho phép MPC bẻ lái gắt ở sát góc cua!

        # ========================================================
        # 2. KHỞI TẠO ROS 2
        # ========================================================
        self.U1 = 0.0 
        self.start_index = None
        self.waypoints = [] 

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.car_frame = "ego_racecar/base_link"
        self.map_frame = "map"

        self.sub_odom = self.create_subscription(Odometry, "ego_racecar/odom", self.odom_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.pub_marker_path = self.create_publisher(MarkerArray, "/publish_full_waypoint", 10)

        csv_path = "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        self.load_waypoints(csv_path)
        self.publish_full_waypoint()
        self.pub_mpc_ref = self.create_publisher(Marker, "/mpc_lookahead_points", 10)

        self.last_mpc_time = self.get_clock().now()
        self.get_logger().info("MPC (Bilinear Stable + Reference Trajectory) đã khởi động!")

    # ========================================================
    # SỬA LỖI TOÁN HỌC: CHUYỂN SANG BILINEAR (TUSTIN)
    # ========================================================
    def calculate_state_space(self, v_x):
        v_x = max(v_x, 1.0) 

        A1 = -(2*self.Caf + 2*self.Car) / (self.m * v_x)
        A2 = -v_x - (2*self.Caf*self.lf - 2*self.Car*self.lr) / (self.m * v_x)
        A3 = -(2*self.lf*self.Caf - 2*self.lr*self.Car) / (self.Iz * v_x)
        A4 = -(2*self.lf**2*self.Caf + 2*self.lr**2*self.Car) / (self.Iz * v_x)

        A_c = np.array([
            [A1, 0, A2, 0],
            [0,  0,  1, 0],
            [A3, 0, A4, 0],
            [1, v_x, 0, 0]
        ])
        B_c = np.array([
            [2*self.Caf / self.m], [0], [2*self.lf*self.Caf / self.Iz], [0]
        ])
        C_c = np.array([
            [0, 1, 0, 0], [0, 0, 0, 1]
        ])

        # CHÌA KHÓA Ở ĐÂY: Dùng nghịch đảo Tustin để đảm bảo ma trận LUÔN STABLE
        I = np.eye(4)
        inv_term = np.linalg.inv(I - (self.Ts / 2.0) * A_c)
        Ad = inv_term @ (I + (self.Ts / 2.0) * A_c)
        Bd = inv_term @ (B_c * self.Ts)
        Cd = C_c
        
        return Ad, Bd, Cd
    

    def mpc_simplification(self, Ad, Bd, Cd):
        A_aug = np.block([[Ad, Bd], [np.zeros((1, 4)), np.eye(1)]])
        B_aug = np.block([[Bd], [np.eye(1)]])
        C_aug = np.block([[Cd, np.zeros((2, 1))]])

        CQC = C_aug.T @ self.Q @ C_aug
        CSC = C_aug.T @ self.S @ C_aug
        QC = self.Q @ C_aug
        SC = self.S @ C_aug

        Qdb = np.zeros((5 * self.hz, 5 * self.hz))
        Tdb = np.zeros((2 * self.hz, 5 * self.hz))
        Rdb = np.zeros((1 * self.hz, 1 * self.hz))
        Cdb = np.zeros((5 * self.hz, 1 * self.hz))
        Adc = np.zeros((5 * self.hz, 5))

        for i in range(self.hz):
            if i == self.hz - 1:
                Qdb[5*i:5*i+5, 5*i:5*i+5] = CSC
                Tdb[2*i:2*i+2, 5*i:5*i+5] = SC
            else:
                Qdb[5*i:5*i+5, 5*i:5*i+5] = CQC
                Tdb[2*i:2*i+2, 5*i:5*i+5] = QC
            Rdb[1*i:1*i+1, 1*i:1*i+1] = self.R
            for j in range(self.hz):
                if j <= i:
                    Cdb[5*i:5*i+5, 1*j:1*j+1] = np.linalg.matrix_power(A_aug, i-j) @ B_aug
            Adc[5*i:5*i+5, :] = np.linalg.matrix_power(A_aug, i+1)

        Hdb = Cdb.T @ Qdb @ Cdb + Rdb
        temp1 = Adc.T @ Qdb @ Cdb
        temp2 = -Tdb @ Cdb
        Fdbt = np.vstack((temp1, temp2))
        return Hdb, Fdbt, Cdb, Adc

    def load_waypoints(self, filename):
        raw_waypoints = []
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row: continue
                    line_data = row[0].split() if len(row) == 1 else row
                    try:
                        raw_waypoints.append([float(line_data[0]), float(line_data[1])])
                    except ValueError: continue
            
            smoothed = self.smooth_path(raw_waypoints)
            self.waypoints = []
            for i in range(len(smoothed)):
                p1 = smoothed[i]
                p2 = smoothed[(i + 1) % len(smoothed)]
                yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                self.waypoints.append([p1[0], p1[1], yaw])
        except Exception as e:
            self.get_logger().error(f"LỖI ĐỌC FILE: {e}")

    def smooth_path(self, path, weight_data=0.5, weight_smooth=0.2, tolerance=0.00001):
        new_path = deepcopy(path)
        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path) - 1):
                aux_x, aux_y = new_path[i][0], new_path[i][1]
                new_path[i][0] += weight_data * (path[i][0] - new_path[i][0]) + \
                                  weight_smooth * (new_path[i-1][0] + new_path[i+1][0] - 2.0 * new_path[i][0])
                new_path[i][1] += weight_data * (path[i][1] - new_path[i][1]) + \
                                  weight_smooth * (new_path[i-1][1] + new_path[i+1][1] - 2.0 * new_path[i][1])
                change += abs(aux_x - new_path[i][0]) + abs(aux_y - new_path[i][1])
        return new_path
    def publish_mpc_reference(self, points_list):
        """Vẽ chuỗi các điểm nhìn trước của MPC lên RViz2"""
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "mpc_reference"
        marker.id = 0
        
        # Dùng SPHERE_LIST để vẽ nhiều quả cầu cùng lúc siêu nhẹ cho CPU
        marker.type = Marker.SPHERE_LIST 
        marker.action = Marker.ADD
        
        # Kích thước quả cầu nhìn trước (15cm)
        marker.scale.x = 0.15 
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        
        # Màu Xanh Lơ (Cyan) nổi bật
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        
        # Nhồi tất cả tọa độ vào Marker
        for pt in points_list:
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = 0.1  # Nâng lên 10cm so với mặt đất cho dễ nhìn
            marker.points.append(p)
            
        self.pub_mpc_ref.publish(marker)

    def odom_callback(self, msg: Odometry):
        if not self.waypoints: return

        # Chặn tần số 20Hz để tránh dồn ứ lệnh
        current_time = self.get_clock().now()
        dt = (current_time - self.last_mpc_time).nanoseconds / 1e9
        if dt < self.Ts: return  
        self.last_mpc_time = current_time

        v_x = msg.twist.twist.linear.x
        v_y = msg.twist.twist.linear.y 
        yaw_rate = msg.twist.twist.angular.z

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.car_frame, rclpy.time.Time(seconds=0))
            rx = transform.transform.translation.x
            ry = transform.transform.translation.y
            q = transform.transform.rotation
            r_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        except TransformException: return

        # TÌM ĐIỂM GẦN NHẤT ĐỂ LÀM GỐC TỌA ĐỘ LOCAL
        min_dist = float("inf")
        nearest_idx = 0
        if self.start_index is None:
            for i, p in enumerate(self.waypoints):
                d = math.hypot(rx - p[0], ry - p[1])
                if d < min_dist:
                    min_dist = d; nearest_idx = i
            self.start_index = nearest_idx
        else:
            nearest_idx = self.start_index
            curr_dist = math.hypot(rx - self.waypoints[nearest_idx][0], ry - self.waypoints[nearest_idx][1])
            for i in range(20): 
                next_idx = (nearest_idx + 1) % len(self.waypoints)
                next_dist = math.hypot(rx - self.waypoints[next_idx][0], ry - self.waypoints[next_idx][1])
                if next_dist < curr_dist:
                    nearest_idx = next_idx
                    curr_dist = next_dist
                else: break
            self.start_index = nearest_idx

        # TÍNH LỖI HIỆN TẠI (Local State)
        wp_x, wp_y, wp_yaw = self.waypoints[nearest_idx]
        e_psi = normalize_angle(r_yaw - wp_yaw)
        dx = rx - wp_x
        dy = ry - wp_y
        e_y = -math.sin(wp_yaw) * dx + math.cos(wp_yaw) * dy

        # CẬP NHẬT MA TRẬN
        Ad, Bd, Cd = self.calculate_state_space(v_x)
        Hdb, Fdbt, Cdb, Adc = self.mpc_simplification(Ad, Bd, Cd)

        states = np.array([v_y, e_psi, yaw_rate, e_y])
        x_aug_t = np.concatenate((states, [self.U1]))
        
        # ========================================================
        # DẠY MPC NHÌN TRƯỚC ĐƯỜNG CUA (Reference Trajectory)
        # ========================================================
        r_list = []
        ref_global_points = []
        
        # Khoảng cách xe sẽ đi được trong 1 bước thời gian Ts
        step_dist = max(v_x, 1.0) * self.Ts 
        
        # Các biến dùng để "trượt" dọc theo đường đua
        curr_idx = nearest_idx
        dist_accum = 0.0
        
        for i in range(1, self.hz + 1):
            target_dist = i * step_dist # Khoảng cách mục tiêu lý tưởng tính từ mũi xe
            
            # Trượt dọc theo các waypoint cho đến khi "kẹp" được cái target_dist vào giữa
            while True:
                next_idx = (curr_idx + 1) % len(self.waypoints)
                p1 = self.waypoints[curr_idx]
                p2 = self.waypoints[next_idx]
                segment_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                
                # Nếu khoảng cách cộng dồn vượt qua target_dist, nghĩa là điểm cần tìm nằm trên đoạn này!
                if dist_accum + segment_len >= target_dist:
                    # Tính tỷ lệ nội suy (ratio) từ 0.0 đến 1.0
                    ratio = (target_dist - dist_accum) / segment_len if segment_len > 0 else 0.0
                    
                    # Nội suy ra tọa độ liên tục (chính xác đến từng milimet)
                    fx = p1[0] + ratio * (p2[0] - p1[0])
                    fy = p1[1] + ratio * (p2[1] - p1[1])
                    
                    # Góc Yaw nội suy chính là vector chỉ phương của đoạn thẳng đó
                    fyaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    break
                else:
                    # Nếu chưa tới, cộng dồn chiều dài và trượt sang waypoint tiếp theo
                    dist_accum += segment_len
                    curr_idx = next_idx
                    
            # Đã tìm được tọa độ liên tục hoàn hảo, đưa vào mảng để vẽ RViz
            ref_global_points.append((fx, fy))
            
            # Tính lỗi TƯƠNG LAI so với GỐC LOCAL (nearest_idx)
            fdx = fx - wp_x
            fdy = fy - wp_y
            local_y = -math.sin(wp_yaw) * fdx + math.cos(wp_yaw) * fdy
            local_yaw = normalize_angle(fyaw - wp_yaw)
            
            # Đẩy vào mảng Reference
            r_list.extend([local_yaw, local_y])
            
        r_vector = np.array(r_list)

        # GIẢI MPC
        self.publish_mpc_reference(ref_global_points)
        ft_input = np.concatenate((x_aug_t, r_vector))
        ft = Fdbt.T @ ft_input
        
        try:
            du = -np.linalg.inv(Hdb) @ ft
            self.U1 = self.U1 + du[0]
            self.U1 = np.clip(self.U1, -0.3, 0.3) 
        except np.linalg.LinAlgError: return

        # XUẤT LỆNH
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(self.U1)

        # Chạy bốc hơn khi đường thẳng, chậm lại lúc vào cua gắt
        if abs(self.U1) > 0.2:
            drive_msg.drive.speed = 3.0 
        else:
            drive_msg.drive.speed = 3.0 
            
        self.pub_drive.publish(drive_msg)

    def publish_full_waypoint(self):
        marker_array = MarkerArray()
        
        # Chỉ tạo ĐÚNG 1 Marker duy nhất
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        
        # Chuyển từ vẽ chấm (SPHERE) sang vẽ đường liên tục (LINE_STRIP)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Độ dày của đường kẻ (0.05 m)
        marker.scale.x = 0.05 
        
        # Đổi sang màu vàng (Yellow) cho nổi bật trên nền xám của Map
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        # Đẩy toàn bộ tọa độ Waypoint vào mảng points của đường kẻ này
        for point in self.waypoints:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.0
            marker.points.append(p)
            
        # [Tùy chọn] Nối điểm cuối với điểm đầu để khép kín vòng đua
        if len(self.waypoints) > 0:
            p_first = Point()
            p_first.x = float(self.waypoints[0][0])
            p_first.y = float(self.waypoints[0][1])
            p_first.z = 0.0
            marker.points.append(p_first)

        marker_array.markers.append(marker)
        self.pub_marker_path.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = MPCNode()
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    except Exception as e: print(f"Lỗi: {e}")
    finally:
        if "node" in locals():
            node.pub_drive.publish(AckermannDriveStamped())
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()