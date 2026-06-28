#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import csv
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.duration import Duration
from copy import deepcopy

# --- HÀM BỔ TRỢ QUAN TRỌNG ---
def normalize_angle(angle):
    """
    Chuẩn hóa góc về khoảng [-pi, pi].
    Ví dụ: 370 độ -> 10 độ.
    Giúp tránh việc xe quay vòng tròn vô lý.
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

class StanleyController(Node):
    def __init__(self):
        super().__init__("stanley_controller_node")
        
        # --- 1. THÔNG SỐ CẤU HÌNH (TUNING PARAMETERS) ---
        self.L = 0.36           # Chiều dài cơ sở (Wheelbase) - Mét
        self.k_gain = 2.4       # Hệ số K (Gain): Càng lớn sửa lỗi càng gắt
        self.k_soft = 1.0       # Hệ số làm mềm: Tránh chia cho 0 khi v=0
        self.window_size = 3    # Kích thước cửa sổ làm mượt hướng (Moving Average)
        
        self.MAX_SPEED = 2.0    # Vận tốc tối đa (m/s)
        self.MAX_STEER = 0.35    # Giới hạn góc lái (rad) (~23 độ)
        
        # --- 2. KHỞI TẠO ROS2 ---
        # Visualization Markers
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.pub_vis_target = self.create_publisher(Marker, "/vis_target_point", 10) 
        self.pub_vis_front = self.create_publisher(Marker, "/vis_front_axle", 10)
        self.pub_vis_path = self.create_publisher(MarkerArray, "/vis_path", 10)
        
        # TF & Odom
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sub_odom = self.create_subscription(Odometry, "odom", self.odom_callback, 10)
        self.car_frame = "base_link"
        self.map_frame = "map"

        # --- 3. LOAD DỮ LIỆU ĐƯỜNG ĐUA ---
        self.waypoints = []
        # !!! THAY ĐỔI ĐƯỜNG DẪN TỚI FILE CSV CỦA BẠN !!!
        csv_path = "/home/danh/ros2_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        self.load_waypoints(csv_path)
        self.publish_path_marker() # Vẽ đường lên Rviz 1 lần
        
        self.get_logger().info(f"Stanley Controller Ready! K={self.k_gain}, Smooth Window={self.window_size}")

    def odom_callback(self, msg: Odometry):
        # --- BƯỚC 1: LẤY VỊ TRÍ XE TRÊN MAP ---
        try:
            # Lấy transform mới nhất từ Map -> Base_link
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, self.car_frame, rclpy.time.Time())
            
            x_base = trans.transform.translation.x
            y_base = trans.transform.translation.y
            
            # Lấy Yaw từ Quaternion
            q = trans.transform.rotation
            _, _, yaw_robot = self.euler_from_quaternion(q.x, q.y, q.z, q.w)
            
            # Lấy vận tốc hiện tại
            current_vel = msg.twist.twist.linear.x

        except TransformException:
            # Nếu chưa có TF thì bỏ qua loop này
            return

        # --- BƯỚC 2: TÍNH VỊ TRÍ TRỤC BÁNH TRƯỚC (QUAN TRỌNG VỚI STANLEY) ---
        # Stanley điều khiển dựa trên bánh trước, không phải tâm xe
        front_x = x_base + self.L * math.cos(yaw_robot)
        front_y = y_base + self.L * math.sin(yaw_robot)
        
        # Visualize trục trước để debug
        self.publish_marker_sphere(self.pub_vis_front, front_x, front_y, 1.0, 0.0, 0.0) # Màu Đỏ

        # --- BƯỚC 3: TÌM ĐIỂM ĐÍCH GẦN NHẤT ---
        target_idx, min_dist = self.get_closest_waypoint_index(front_x, front_y)
        
        # Visualize điểm đích
        tx = self.waypoints[target_idx][0]
        ty = self.waypoints[target_idx][1]
        self.publish_marker_sphere(self.pub_vis_target, tx, ty, 0.0, 1.0, 0.0) # Màu Xanh Lá

        # --- BƯỚC 4: TÍNH LỖI HƯỚNG (HEADING ERROR - PSI_E) ---
        # Sử dụng hàm làm mượt (Smooth Yaw) thay vì chỉ lấy 2 điểm
        track_yaw = self.calculate_smooth_yaw(target_idx, self.window_size)
        
        psi_e = normalize_angle(track_yaw - yaw_robot)

        # --- BƯỚC 5: TÍNH LỖI VỊ TRÍ (CROSS TRACK ERROR - CTE) ---
        # CTE là khoảng cách từ trục trước đến đường.
        # Cần xác định dấu: Xe đang lệch trái hay lệch phải?
        
        # Vector đường đi
        path_vector_x = math.cos(track_yaw)
        path_vector_y = math.sin(track_yaw)
        
        # Vector từ đường đến xe (Front axle)
        dx = front_x - tx
        dy = front_y - ty
        
        # Tích có hướng (Cross Product) 2D để tìm phía
        # Cross = dx * dy_path - dy * dx_path (hoặc ngược lại tùy hệ quy chiếu)
        # Ở đây ta dùng công thức: (Xe - Đường) x (Hướng Đường)
        cross_product = dx * path_vector_y - dy * path_vector_x
        
        # Quy ước: Nếu xe lệch trái đường, ta cần đánh lái sang phải (CTE dương -> Steer âm)
        # Trong hệ tọa độ này: 
        # Nếu cross_product > 0: Xe đang ở bên PHẢI đường (so với hướng đi tới)
        # Nếu cross_product < 0: Xe đang ở bên TRÁI đường
        
        cte = min_dist
        if cross_product > 0:
            cte = -cte # Xe bên phải, CTE âm
        else:
            cte = cte  # Xe bên trái, CTE dương

        # --- BƯỚC 6: CÔNG THỨC STANLEY ---
        # Delta = Psi_e + arctan(k * e / (v + k_soft))
        
        # Lưu ý dấu: Nếu xe lệch trái (CTE > 0), ta cần lái về bên phải (Góc lái âm).
        # Công thức chuẩn: angle = psi_e - arctan(...) nếu định nghĩa CTE như trên.
        # Hoặc cộng đại số: angle = psi_e + arctan(-k * cte / ...)
        
        # Hãy dùng logic này:
        # 1. Hướng sai: psi_e dương (lệch trái) -> cần giảm góc lái -> trừ hoặc cộng tuỳ hướng.
        # Logic chuẩn ROS (X tới, Y trái, Z lên): 
        # Góc dương là quay trái.
        # Nếu lệch trái (psi_e > 0) -> Cần quay phải (Steer < 0) -> Steer = -psi_e ... (cần cẩn thận)
        
        # Ta dùng công thức chuẩn tắc Stanley:
        # delta = psi_e + arctan( k * e / v )
        # Trong đó e là khoảng cách CÓ DẤU.
        # Nếu e > 0 (xe lệch trái tim đường) -> cần lái phải (âm). Vây term sau phải ra âm.
        # -> delta = psi_e + arctan( -k * abs(e) / v )  (nếu lệch trái)
        
        # Rút gọn lại theo biến `cte` đã tính dấu ở trên:
        # Nếu lệch trái (cte > 0) -> cần lái phải -> term arctan phải âm. -> dùng -k
        steering_angle = psi_e + math.atan2(-self.k_gain * cte, current_vel + self.k_soft)
        
        # Giới hạn góc lái (Saturation)
        steering_angle = np.clip(steering_angle, -self.MAX_STEER, self.MAX_STEER)

        # --- BƯỚC 7: GỬI LỆNH LÁI ---
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        
        # Logic tốc độ đơn giản: Cua gắt thì giảm tốc
        if abs(steering_angle) > 0.15:
            drive_msg.drive.speed = 1.9
        

        else:
            drive_msg.drive.speed = self.MAX_SPEED
            
        self.pub_drive.publish(drive_msg)

        # Debug log (Optional - bỏ comment nếu muốn xem số liệu)
        # self.get_logger().info(f"CTE: {cte:.2f} | Psi_e: {psi_e:.2f} | Steer: {steering_angle:.2f}")

    # --- HÀM TÍNH TOÁN LOGIC ---
    
    def calculate_smooth_yaw(self, current_idx, window):
        """
        Tính hướng trung bình của 'window' điểm phía trước.
        Dùng sin/cos để tránh lỗi đứt gãy góc (ví dụ 179 độ và -179 độ).
        """
        sum_sin = 0.0
        sum_cos = 0.0
        
        for i in range(window):
            # Lấy chỉ số vòng tròn (Circular buffer)
            idx1 = (current_idx + i) % len(self.waypoints)
            idx2 = (current_idx + i + 1) % len(self.waypoints)
            
            p1 = self.waypoints[idx1]
            p2 = self.waypoints[idx2]
            
            # Góc của đoạn nhỏ này
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            
            sum_sin += math.sin(angle)
            sum_cos += math.cos(angle)
            
        # Tính lại góc từ tổng vector
        return math.atan2(sum_sin, sum_cos)

    def get_closest_waypoint_index(self, x, y):
        # Tìm khoảng cách Euclid từ (x,y) đến TẤT CẢ các điểm (có thể tối ưu sau)
        dists = [math.hypot(p[0] - x, p[1] - y) for p in self.waypoints]
        min_dist = min(dists)
        idx = dists.index(min_dist)
        return idx, min_dist

    # --- CÁC HÀM XỬ LÝ DỮ LIỆU & VISUALIZATION ---
    
    def load_waypoints(self, file_path):
        raw_points = []
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Xử lý format file csv của bạn
                    if not row: continue
                    try:
                        line = row[0].split() if len(row) == 1 else row
                        x = float(line[0])
                        y = float(line[1])
                        raw_points.append([x, y])
                    except ValueError: continue
            
            # Làm mượt đường đi (Smooth Path - Gradient Descent) trước khi dùng
            self.waypoints = self.smooth_path(raw_points)
            self.get_logger().info(f"Loaded {len(self.waypoints)} points.")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load CSV: {e}")

    def smooth_path(self, path, weight_data=0.5, weight_smooth=0.5, tolerance=0.00001):
        # Hàm làm mượt đường đua (giữ nguyên từ code cũ của bạn)
        new_path = deepcopy(path)
        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path)-1):
                aux_x = new_path[i][0]
                aux_y = new_path[i][1]
                new_path[i][0] += weight_data * (path[i][0] - new_path[i][0]) + \
                                  weight_smooth * (new_path[i-1][0] + new_path[i+1][0] - 2.0 * new_path[i][0])
                new_path[i][1] += weight_data * (path[i][1] - new_path[i][1]) + \
                                  weight_smooth * (new_path[i-1][1] + new_path[i+1][1] - 2.0 * new_path[i][1])
                change += abs(aux_x - new_path[i][0]) + abs(aux_y - new_path[i][1])
        return new_path

    def euler_from_quaternion(self, x, y, z, w):
        # Hàm chuyển đổi TF chuẩn
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        return 0, 0, math.atan2(t3, t4) # Chỉ cần trả về Yaw

    def publish_marker_sphere(self, publisher, x, y, r, g, b):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.2; marker.scale.y = 0.2; marker.scale.z = 0.2
        marker.color.a = 1.0; marker.color.r = r; marker.color.g = g; marker.color.b = b
        marker.pose.position.x = x; marker.pose.position.y = y
        publisher.publish(marker)

    def publish_path_marker(self):
        # Vẽ toàn bộ đường đi 1 lần
        ma = MarkerArray()
        for i, p in enumerate(self.waypoints):
            m = Marker()
            m.header.frame_id = "map"
            m.id = i
            m.type = Marker.SPHERE
            m.scale.x = 0.05; m.scale.y = 0.05; m.scale.z = 0.05
            m.color.a = 1.0; m.color.r = 0.7; m.color.g = 0.7; m.color.b = 0.7
            m.pose.position.x = p[0]; m.pose.position.y = p[1]
            ma.markers.append(m)
        self.pub_vis_path.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = StanleyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()