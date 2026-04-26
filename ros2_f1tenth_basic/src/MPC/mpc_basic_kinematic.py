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
from geometry_msgs.msg import Point

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

def normalize_angle(angle):
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle

class KinematicMPCNode(Node):
    def __init__(self):
        super().__init__("kinematic_mpc_node")

        # ========================================================
        # 1. THAM SỐ KINEMATIC CHO F1TENTH THỰC TẾ
        # ========================================================
        self.lf = 0.158     
        self.lr = 0.171     
        self.L = self.lf + self.lr  # Chiều dài cơ sở (Wheelbase)
        self.Ts = 0.05              # Lấy mẫu 20Hz
        
        self.hz = 30                # Số bước nhìn trước (Horizon)
        
        # Ma trận trọng số Q (Phạt sai số) và R (Phạt đánh lái)
        # Hệ chỉ còn 2 trạng thái: [e_y, e_psi]
        self.Q = np.array([[150.0, 0.0],   # Phạt lệch quỹ đạo e_y
                           [0.0, 50.0]])   # Phạt lệch góc ngắm e_psi
                           
        self.S = np.array([[150.0, 0.0],   # Phạt nặng điểm cuối cùng (Terminal Cost)
                           [0.0, 80.0]])  
                           
        self.R = np.array([[400.0]])       # Phạt tốc độ bẻ lái (Delta Steering)

        # ========================================================
        # 2. KHỞI TẠO ROS 2 CHO XE THẬT
        # ========================================================
        self.U1 = 0.0 
        self.start_index = None
        self.waypoints = [] 

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Đổi frame cho phù hợp với xe thật
        self.car_frame = "base_link" 
        self.map_frame = "map"

        # Đổi topic odometry chuẩn
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        
        # Topic điều khiển (Thường xe thật dùng mux, ví dụ: /vesc/high_level/...)
        # Tạm thời để /drive, bạn có thể remap khi launch
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        
        self.pub_marker_path = self.create_publisher(MarkerArray, "/publish_full_waypoint", 10)
        self.pub_mpc_ref = self.create_publisher(Marker, "/mpc_lookahead_points", 10)

        # Cập nhật đường dẫn file csv phù hợp với file hệ thống trên xe
        csv_path = "/home/danh/ros2_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv" # Yêu cầu truyền đường dẫn tuyệt đối hoặc dùng ament_index_python
        self.load_waypoints(csv_path)
        self.publish_full_waypoint()

        self.last_mpc_time = self.get_clock().now()
        self.get_logger().info("Kinematic MPC đã khởi động! Sẵn sàng chạy xe thật.")

    # ========================================================
    # MÔ HÌNH KINEMATIC ERROR RỜI RẠC HÓA CHÍNH XÁC (ZOH)
    # ========================================================
    def calculate_kinematic_state_space(self, v_x):
        # Tránh lỗi chia cho 0 khi xe đứng im
        v_x = max(v_x, 0.5) 

        # Ma trận rời rạc hóa (Zero-Order Hold) cho mô hình Kinematic Error
        # Trạng thái x = [e_y, e_psi]^T
        # Đầu vào u = steering_angle
        
        Ad = np.array([
            [1.0, v_x * self.Ts],
            [0.0, 1.0]
        ])
        
        Bd = np.array([
            [(v_x**2 * self.Ts**2) / (2 * self.L)],
            [(v_x * self.Ts) / self.L]
        ])
        
        Cd = np.eye(2)
        
        return Ad, Bd, Cd

    def mpc_simplification(self, Ad, Bd, Cd):
        # Thiết lập mô hình Augmented (Delta u formulation)
        # x_aug = [e_y, e_psi, u_k-1]^T
        A_aug = np.block([[Ad, Bd], [np.zeros((1, 2)), np.eye(1)]])
        B_aug = np.block([[Bd], [np.eye(1)]])
        C_aug = np.block([[Cd, np.zeros((2, 1))]])

        CQC = C_aug.T @ self.Q @ C_aug
        CSC = C_aug.T @ self.S @ C_aug
        QC = self.Q @ C_aug
        SC = self.S @ C_aug

        # Kích thước ma trận giảm mạnh từ (5*hz) xuống còn (3*hz)
        Qdb = np.zeros((3 * self.hz, 3 * self.hz))
        Tdb = np.zeros((2 * self.hz, 3 * self.hz))
        Rdb = np.zeros((1 * self.hz, 1 * self.hz))
        Cdb = np.zeros((3 * self.hz, 1 * self.hz))
        Adc = np.zeros((3 * self.hz, 3))

        for i in range(self.hz):
            if i == self.hz - 1:
                Qdb[3*i:3*i+3, 3*i:3*i+3] = CSC
                Tdb[2*i:2*i+2, 3*i:3*i+3] = SC
            else:
                Qdb[3*i:3*i+3, 3*i:3*i+3] = CQC
                Tdb[2*i:2*i+2, 3*i:3*i+3] = QC
            Rdb[i, i] = self.R[0, 0]
            
            for j in range(self.hz):
                if j <= i:
                    Cdb[3*i:3*i+3, j:j+1] = np.linalg.matrix_power(A_aug, i-j) @ B_aug
            Adc[3*i:3*i+3, :] = np.linalg.matrix_power(A_aug, i+1)

        Hdb = Cdb.T @ Qdb @ Cdb + Rdb
        Fdbt = np.vstack((Adc.T @ Qdb @ Cdb, -Tdb @ Cdb))
        return Hdb, Fdbt, Cdb, Adc

    # ... (Các hàm load_waypoints, smooth_path, publish_mpc_reference, publish_full_waypoint giữ nguyên) ...

    def odom_callback(self, msg: Odometry):
        if not self.waypoints: return

        current_time = self.get_clock().now()
        dt = (current_time - self.last_mpc_time).nanoseconds / 1e9
        if dt < self.Ts: return  
        self.last_mpc_time = current_time

        # Trên xe thật, thường chỉ tin tưởng v_x từ Odom, v_y coi như = 0
        v_x = msg.twist.twist.linear.x

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.car_frame, rclpy.time.Time())
            rx = transform.transform.translation.x
            ry = transform.transform.translation.y
            q = transform.transform.rotation
            r_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        except TransformException: return

        # TÌM ĐIỂM GẦN NHẤT
        min_dist = float("inf")
        nearest_idx = self.start_index if self.start_index is not None else 0
        
        search_range = range(len(self.waypoints)) if self.start_index is None else range(nearest_idx, nearest_idx + 20)
        for i_raw in search_range:
            i = i_raw % len(self.waypoints)
            d = math.hypot(rx - self.waypoints[i][0], ry - self.waypoints[i][1])
            if d < min_dist:
                min_dist = d
                self.start_index = i

        nearest_idx = self.start_index
        wp_x, wp_y, wp_yaw = self.waypoints[nearest_idx]

        # TÍNH LỖI KINEMATIC HIỆN TẠI (e_y, e_psi)
        dx = rx - wp_x
        dy = ry - wp_y
        e_y = -math.sin(wp_yaw) * dx + math.cos(wp_yaw) * dy
        e_psi = normalize_angle(r_yaw - wp_yaw)

        # CẬP NHẬT MA TRẬN KINEMATIC
        Ad, Bd, Cd = self.calculate_kinematic_state_space(v_x)
        Hdb, Fdbt, Cdb, Adc = self.mpc_simplification(Ad, Bd, Cd)

        # STATE VECTOR GỌN NHẸ: Chỉ còn [e_y, e_psi, U_prev]
        states = np.array([e_y, e_psi])
        x_aug_t = np.concatenate((states, [self.U1]))
        
        # TẠO REFERENCE TRAJECTORY (Chỉ tạo cho e_y và e_psi tương lai)
        r_list = []
        ref_global_points = []
        step_dist = max(v_x, 0.5) * self.Ts 
        curr_idx = nearest_idx
        dist_accum = 0.0
        
        for i in range(1, self.hz + 1):
            target_dist = i * step_dist 
            while True:
                next_idx = (curr_idx + 1) % len(self.waypoints)
                p1 = self.waypoints[curr_idx]
                p2 = self.waypoints[next_idx]
                segment_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                
                if dist_accum + segment_len >= target_dist:
                    ratio = (target_dist - dist_accum) / segment_len if segment_len > 0 else 0.0
                    fx = p1[0] + ratio * (p2[0] - p1[0])
                    fy = p1[1] + ratio * (p2[1] - p1[1])
                    fyaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    break
                else:
                    dist_accum += segment_len
                    curr_idx = next_idx
                    
            ref_global_points.append((fx, fy))
            
            # Tính e_y và e_psi của tương lai so với gốc tọa độ LOCAL hiện tại
            fdx = fx - wp_x
            fdy = fy - wp_y
            future_e_y = -math.sin(wp_yaw) * fdx + math.cos(wp_yaw) * fdy
            future_e_psi = normalize_angle(fyaw - wp_yaw)
            
            r_list.extend([future_e_y, future_e_psi])
            
        r_vector = np.array(r_list)

        # GIẢI MPC (Cực kỳ nhanh vì ma trận Hdb giờ rất nhỏ)
        self.publish_mpc_reference(ref_global_points)
        ft_input = np.concatenate((x_aug_t, r_vector))
        ft = Fdbt.T @ ft_input
        
        try:
            # Nghịch đảo Hdb giờ đây nhẹ hơn hàng chục lần so với hệ Dynamic
            du = -np.linalg.inv(Hdb) @ ft
            self.U1 = self.U1 + du[0]
            
            # Giới hạn góc lái vật lý của F1TENTH (khoảng 20-25 độ)
            max_steer = 0.35 
            self.U1 = np.clip(self.U1, -max_steer, max_steer) 
        except np.linalg.LinAlgError: return

        # ========================================================
        # LOGIC AN TOÀN TỐC ĐỘ (SPEED PROFILING) CHO XE THẬT
        # ========================================================
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(self.U1)

        # KHÔNG ĐƯỢC DÙNG 7.5 m/s. Xe thật sẽ mất kiểm soát ngay lập tức!
        # Tốc độ tỉ lệ nghịch với độ lớn của góc bẻ lái: 
        # Càng bẻ lái gắt -> Càng đi chậm lại.
        base_speed = 1.6  # Tốc độ tối đa trên đường thẳng (m/s)
        min_speed = 1.6   # Tốc độ tối thiểu khi ôm cua gắt (m/s)
        
        # Công thức phanh mượt mà khi vào cua
        target_speed = base_speed - (abs(self.U1) / max_steer) * (base_speed - min_speed)
        drive_msg.drive.speed = max(min_speed, min(base_speed, target_speed))
            
        self.pub_drive.publish(drive_msg)
        
    # (Hàm load_waypoints, publish... giữ nguyên)
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

def main(args=None):
    rclpy.init(args=args)
    try:
        node = KinematicMPCNode()
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if "node" in locals():
            node.pub_drive.publish(AckermannDriveStamped()) # Gửi lệnh dừng xe
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()