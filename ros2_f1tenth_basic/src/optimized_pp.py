#!/usr/bin/env python3
"""
Tên Node: optimized_pure_pursuit_node
Mô tả: Node điều khiển bám quỹ đạo Pure Pursuit tối ưu hóa cho ROS 2.
Các tính năng chính:
- Adaptive Lookahead dựa trên vận tốc.
- Velocity Profiling dựa trên độ cong quỹ đạo (Vector hóa NumPy).
- Xử lý TF2 an toàn (Non-blocking, Time 0).
- Tính toán khoảng cách và tìm kiếm điểm mục tiêu bằng NumPy Broadcasting.
- Cấu hình động qua ROS 2 Parameters.

Tác giả: Chuyên gia Hệ thống Tự hành
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

import numpy as np
import math

# Import các loại message cần thiết
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header

# Import thư viện TF2 để xử lý biến đổi tọa độ
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf_transformations import euler_from_quaternion

class OptimizedPurePursuit(Node):
    def __init__(self):
        super().__init__('optimized_pure_pursuit')

        # ========================================================================================
        # 1. KHỞI TẠO THAM SỐ (ROS 2 PARAMETERS)
        # ========================================================================================
        # Khai báo tham số với giá trị mặc định. Người dùng có thể ghi đè qua launch file hoặc CLI.
        
        # Thông số vật lý của robot
        self.declare_parameter('wheelbase', 0.33)  # Chiều dài cơ sở (m) - Ví dụ cho xe F1Tenth
        
        # Giới hạn động học
        self.declare_parameter('max_velocity', 2.0)      # Vận tốc tối đa (m/s)
        self.declare_parameter('min_velocity', 0.5)      # Vận tốc tối thiểu khi vào cua (m/s)
        self.declare_parameter('max_steering_angle', 0.5) # Góc lái tối đa (rad) ~ 28 độ
        
        # Thuật toán Adaptive Lookahead
        self.declare_parameter('lookahead_time', 0.1)    # Hệ số thời gian nhìn trước (s)
        self.declare_parameter('min_lookahead_dist', 0.5) # Khoảng cách nhìn trước tối thiểu (m)
        self.declare_parameter('max_lookahead_dist', 2.0) # Khoảng cách nhìn trước tối đa (m)
        
        # Thuật toán Velocity Profiling
        self.declare_parameter('velocity_curvature_gain', 0.8) # Hệ số giảm tốc theo độ cong
        
        # Cấu hình Topic và Frame
        self.declare_parameter('odom_topic', '/odometry')
        self.declare_parameter('path_topic', '/plan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('global_frame', 'map')       # Frame của bản đồ/quỹ đạo
        self.declare_parameter('robot_frame', 'base_link')  # Frame của robot

        # Lấy giá trị tham số
        self.wheelbase = self.get_parameter('wheelbase').value
        self.max_vel = self.get_parameter('max_velocity').value
        self.min_vel = self.get_parameter('min_velocity').value
        self.max_steer = self.get_parameter('max_steering_angle').value
        self.lookahead_time = self.get_parameter('lookahead_time').value
        self.min_lookahead = self.get_parameter('min_lookahead_dist').value
        self.max_lookahead = self.get_parameter('max_lookahead_dist').value
        self.vel_curve_gain = self.get_parameter('velocity_curvature_gain').value
        
        self.global_frame = self.get_parameter('global_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value

        # ========================================================================================
        # 2. THIẾT LẬP HẠ TẦNG (TF2, PUBS/SUBS, QOS)
        # ========================================================================================
        
        # Buffer lưu trữ các transform. Cần thiết để biến đổi tọa độ điểm mục tiêu về hệ quy chiếu robot.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Biến trạng thái
        self.path_array = None         # Mảng NumPy chứa tọa độ (N, 2)
        self.curvature_profile = None  # Mảng NumPy chứa độ cong tại mỗi điểm (N,)
        self.latest_odom = None
        self.current_speed = 0.0
        
        # Cấu hình QoS 
        # Odom: Best Effort để giảm độ trễ cho dữ liệu thời gian thực
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Path: Transient Local để nhận được map/path đã publish trước khi node này khởi động
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.create_subscription(Path, self.get_parameter('path_topic').value, self.path_callback, path_qos)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.odom_callback, odom_qos)
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.debug_pub = self.create_publisher(PointStamped, '/lookahead_point', 10)
        
        # Timer điều khiển chính (50 Hz - Chu kỳ 20ms)
        self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info("Optimized Pure Pursuit Node đã khởi động thành công.")

    # ========================================================================================
    # 3. XỬ LÝ QUỸ ĐẠO VÀ TÍNH TOÁN ĐỘ CONG (VECTOR HÓA)
    # ========================================================================================
    def path_callback(self, msg: Path):
        """
        Callback nhận quỹ đạo toàn cục. 
        Thực hiện tiền xử lý (pre-processing) để tính độ cong ngay lập tức bằng NumPy.
        """
        if not msg.poses:
            self.get_logger().warn("Nhận được quỹ đạo rỗng!")
            return

        # Chuyển đổi list ROS messages thành mảng NumPy (N, 2) để tính toán SIMD
        # Việc này nhanh hơn hàng chục lần so với list comprehension trong vòng lặp điều khiển
        path_coords = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        
        # ---------------------------------------------------------------
        # Tính độ cong (Curvature) sử dụng Sai phân trung tâm (Central Difference)
        # Công thức: k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        # ---------------------------------------------------------------
        
        # Đạo hàm bậc 1 (Vận tốc biến thiên theo chỉ số)
        dx = np.gradient(path_coords[:, 0])
        dy = np.gradient(path_coords[:, 1])
        
        # Đạo hàm bậc 2 (Gia tốc biến thiên theo chỉ số)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Tính độ cong
        # Thêm epsilon (1e-9) để tránh lỗi chia cho 0 khi đi đường thẳng (độ cong = 0)
        denominator = np.power(dx**2 + dy**2, 1.5) + 1e-9
        numerator = np.abs(dx * ddy - dy * ddx)
        curvature = numerator / denominator
        
        # Lưu trữ dữ liệu đã xử lý
        self.path_array = path_coords
        self.curvature_profile = curvature
        self.path_frame_id = msg.header.frame_id
        
        self.get_logger().info(f"Đã xử lý quỹ đạo: {len(self.path_array)} điểm.")

    def odom_callback(self, msg: Odometry):
        """Cập nhật trạng thái robot từ Odometry."""
        self.latest_odom = msg
        # Tính vận tốc tuyến tính tổng hợp
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)

    # ========================================================================================
    # 4. VÒNG LẶP ĐIỀU KHIỂN CHÍNH (CONTROL LOOP)
    # ========================================================================================
    def control_loop(self):
        """
        Thực thi thuật toán Pure Pursuit.
        Các bước: TF Lookup -> Tìm điểm gần nhất -> Tính Lookahead -> Tìm đích -> Tính lái -> Gửi lệnh.
        """
        if self.path_array is None or self.latest_odom is None:
            return

        # ---------------------------------------------------------------
        # Bước 1: Xác định vị trí Robot an toàn với TF2
        # ---------------------------------------------------------------
        try:
            # Sử dụng Time() thay vì Time.now() để lấy transform mới nhất có sẵn
            # Tránh lỗi ExtrapolationException do độ trễ mạng [8, 21]
            trans = self.tf_buffer.lookup_transform(
                self.path_frame_id,
                self.robot_frame,
                rclpy.time.Time())
            
            robot_x = trans.transform.translation.x
            robot_y = trans.transform.translation.y
            
            # Chuyển đổi Quaternion sang Euler để lấy góc hướng (Yaw)
            q = trans.transform.rotation
            _, _, robot_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # Chỉ cảnh báo (warn) để không làm spam log, sử dụng throttle
            self.get_logger().warn(f"Lỗi TF: {e}", throttle_duration_sec=1.0)
            return

        # ---------------------------------------------------------------
        # Bước 2: Tìm điểm gần nhất (Nearest Neighbor) bằng NumPy Broadcasting
        # ---------------------------------------------------------------
        # Tính khoảng cách từ robot đến TẤT CẢ các điểm trong quỹ đạo cùng lúc
        # Kỹ thuật Broadcasting: (N, 2) - (1, 2)
        robot_pos = np.array([robot_x, robot_y])
        deltas = self.path_array - robot_pos
        distances = np.hypot(deltas[:, 0], deltas[:, 1]) # Tính cạnh huyền (Euclidean dist)
        
        # Lấy chỉ số (index) của điểm có khoảng cách nhỏ nhất
        nearest_index = np.argmin(distances)
        
        # ---------------------------------------------------------------
        # Bước 3: Tính khoảng cách nhìn trước thích ứng (Adaptive Lookahead) [7]
        # ---------------------------------------------------------------
        # Lookahead tăng theo vận tốc để ổn định xe
        adaptive_lookahead = np.clip(
            self.current_speed * self.lookahead_time, 
            self.min_lookahead, 
            self.max_lookahead
        )
        
        # ---------------------------------------------------------------
        # Bước 4: Tìm điểm mục tiêu (Lookahead Point)
        # ---------------------------------------------------------------
        # Tìm điểm đầu tiên trên quỹ đạo (phía trước điểm gần nhất) thỏa mãn khoảng cách >= adaptive_lookahead
        
        # Cắt mảng khoảng cách từ vị trí gần nhất trở đi
        future_distances = distances[nearest_index:]
        
        # Tìm các chỉ số thỏa mãn điều kiện
        candidates = np.where(future_distances >= adaptive_lookahead)
        
        if len(candidates) > 0:
            # Cộng lại offset để ra chỉ số toàn cục
            target_index = nearest_index + candidates
        else:
            # Nếu không có điểm nào đủ xa (đang ở cuối đường), chọn điểm cuối cùng
            target_index = len(self.path_array) - 1
            
        target_point = self.path_array[target_index]
        
        # ---------------------------------------------------------------
        # Bước 5: Tính góc lái (Geometric Pure Pursuit Control Law)
        # ---------------------------------------------------------------
        
        # Vector từ robot đến mục tiêu
        dx = target_point - robot_x
        dy = target_point[1] - robot_y
        
        # Góc của vector mục tiêu trong hệ tọa độ map
        angle_to_target = math.atan2(dy, dx)
        
        # Góc alpha: Góc lệch giữa hướng xe và hướng mục tiêu
        alpha = angle_to_target - robot_yaw
        
        # Chuẩn hóa góc về đoạn [-pi, pi] để tránh lỗi quay vòng
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        
        # Khoảng cách thực tế đến điểm mục tiêu (L_d)
        L_d = math.hypot(dx, dy)
        
        # Công thức Pure Pursuit: delta = atan(2 * L * sin(alpha) / L_d)
        # 
        if L_d > 0.01: 
            steering_angle = math.atan2(2 * self.wheelbase * math.sin(alpha), L_d)
        else:
            steering_angle = 0.0
            
        # Giới hạn góc lái bão hòa (Saturation)
        steering_angle = max(min(steering_angle, self.max_steer), -self.max_steer)
        
        # ---------------------------------------------------------------
        # Bước 6: Velocity Profiling (Dựa trên độ cong) 
        # ---------------------------------------------------------------
        # Tra cứu độ cong tại điểm mục tiêu đã tính sẵn
        target_curvature = self.curvature_profile[target_index]
        
        # Công thức giảm tốc heuristic: v = v_max / (1 + K * |curvature|)
        # Khúc cua càng gắt (curvature lớn) -> Vận tốc càng giảm
        target_velocity = self.max_vel / (1.0 + self.vel_curve_gain * target_curvature)
        
        # Đảm bảo không thấp hơn vận tốc tối thiểu
        target_velocity = max(target_velocity, self.min_vel)
        
        # ---------------------------------------------------------------
        # Bước 7: Xuất lệnh điều khiển
        # ---------------------------------------------------------------
        twist = Twist()
        twist.linear.x = float(target_velocity)
        
        # Chuyển đổi góc lái bánh trước (delta) sang vận tốc góc (omega) cho cmd_vel
        # Mô hình xe đạp: omega = (v / L) * tan(delta)
        twist.angular.z = (target_velocity / self.wheelbase) * math.tan(steering_angle)
        
        self.cmd_pub.publish(twist)
        
        # Debug: Hiển thị điểm đang nhắm tới trên Rviz
        self.publish_debug_point(target_point)

    def publish_debug_point(self, point):
        """Helper để visualize điểm mục tiêu trên Rviz."""
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.global_frame
        msg.point.x = point
        msg.point.y = point[1]
        self.debug_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = OptimizedPurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # An toàn: Gửi lệnh dừng xe khi tắt node
        stop_msg = Twist()
        node.cmd_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()