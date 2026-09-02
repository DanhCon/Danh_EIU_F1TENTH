#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
import math

class SequenceMotionNode(Node):
    def __init__(self):
        super().__init__('sequence_motion_node')

        # ------------------- THAM SỐ VẬN TỐC -------------------
        self.declare_parameter('max_linear_vel', 0.25)    # Tốc độ tiến/lùi tối đa (m/s)
        self.declare_parameter('min_linear_vel', 0.05)    # Tốc độ tiến/lùi tối thiểu
        self.declare_parameter('max_angular_vel', 0.4)    # Tốc độ quay tối đa (rad/s)
        self.declare_parameter('min_angular_vel', 0.12)   # Tốc độ quay tối thiểu
        self.declare_parameter('kp_linear', 1.0)          # Hệ số P cho khoảng cách
        self.declare_parameter('kp_angular', 1.2)         # Hệ số P cho góc quay
        self.declare_parameter('auto_start', True)        # Tự chạy ngay khi bật Node

        self.max_v = self.get_parameter('max_linear_vel').value
        self.min_v = self.get_parameter('min_linear_vel').value
        self.max_w = self.get_parameter('max_angular_vel').value
        self.min_w = self.get_parameter('min_angular_vel').value
        self.kp_lin = self.get_parameter('kp_linear').value
        self.kp_ang = self.get_parameter('kp_angular').value
        self.auto_start = self.get_parameter('auto_start').value

        # ------------------- DEFINITION CHUỖI HÀNH ĐỘNG -------------------
        self.sequence = [
            {"type": "FORWARD",  "dist": 1.0,  "desc": "Bước 1: Tiến thẳng 1.0m"},
            {"type": "PAUSE",    "time": 2.0,  "desc": "Tạm nghỉ 2.0s"},
            {"type": "BACKWARD", "dist": 1.0,  "desc": "Bước 2: Lùi lại 1.0m"},
            {"type": "PAUSE",    "time": 2.0,  "desc": "Tạm nghỉ 2.0s"},
            {"type": "TURN",     "deg": 90.0,  "desc": "Bước 3: Quay sang trái 90°"},
            {"type": "PAUSE",    "time": 2.0,  "desc": "Tạm nghỉ 2.0s"},
            {"type": "TURN",     "deg": -90.0, "desc": "Bước 4: Quay lại thẳng (Quay phải 90°)"},
            {"type": "PAUSE",    "time": 2.0,  "desc": "Tạm nghỉ 2.0s"},
            {"type": "TURN",     "deg": -90.0, "desc": "Bước 5: Quay sang phải 90°"},
            {"type": "PAUSE",    "time": 2.0,  "desc": "Tạm nghỉ 2.0s"},
            {"type": "TURN",     "deg": 90.0,  "desc": "Bước 6: Quay lại thẳng (Quay trái 90°)"},
        ]

        # ------------------- PUBLISHER & SUBSCRIBER -------------------
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.srv = self.create_service(Trigger, 'start_sequence', self.trigger_start_callback)

        # Timer vòng lặp điều khiển 20Hz (mỗi 50ms)
        self.control_timer = self.create_timer(0.05, self.control_loop)

        # ------------------- BIẾN QUẢN LÝ TRẠNG THÁI -------------------
        self.current_step_idx = -1
        self.is_running = False
        self.odom_received = False

        # Dữ liệu vị trí / góc hiện tại từ Odom
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Dữ liệu mốc bắt đầu của mỗi bước
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0
        self.last_yaw = 0.0
        self.accumulated_yaw = 0.0
        self.pause_start_time = None

        self.get_logger().info("=== CHUỖI HÀNH ĐỘNG (SEQUENCE MOTION NODE) ĐÃ KHỞI ĐỘNG ===")
        if self.auto_start:
            self.start_sequence()

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle_diff(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def start_sequence(self):
        self.current_step_idx = 0
        self.is_running = True
        self.init_step(self.current_step_idx)

    def trigger_start_callback(self, request, response):
        self.start_sequence()
        response.success = True
        response.message = "Đã khởi chạy lại chuỗi hành động."
        return response

    def init_step(self, idx):
        """ Khởi tạo mốc dữ liệu ban đầu cho bước hành động thứ idx """
        if idx >= len(self.sequence):
            self.is_running = False
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            self.get_logger().info("==================================================")
            self.get_logger().info("🎉 CHÚC MỪNG! ĐÃ HOÀN THÀNH TOÀN BỘ CHUỖI HÀNH ĐỘNG!")
            self.get_logger().info("==================================================")
            return

        step = self.sequence[idx]
        self.get_logger().info(f"▶️ BẮT ĐẦU: {step['desc']}")
        
        self.start_x = self.current_x
        self.start_y = self.current_y
        self.start_yaw = self.current_yaw
        self.last_yaw = self.current_yaw
        self.accumulated_yaw = 0.0
        self.pause_start_time = self.get_clock().now()

    def odom_callback(self, msg: Odometry):
        self.odom_received = True
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = self.get_yaw_from_quaternion(msg.pose.pose.orientation)

        # Đọc cộng dồn góc khi đang thực hiện bước TURN
        if self.is_running and self.current_step_idx < len(self.sequence):
            if self.sequence[self.current_step_idx]["type"] == "TURN":
                d_yaw = self.normalize_angle_diff(self.current_yaw - self.last_yaw)
                self.accumulated_yaw += d_yaw
                self.last_yaw = self.current_yaw

    def control_loop(self):
        if not self.is_running or self.current_step_idx >= len(self.sequence):
            return

        if not self.odom_received:
            self.get_logger().warn("Đang chờ dữ liệu /odom...", throttle_duration_sec=2.0)
            return

        step = self.sequence[self.current_step_idx]
        step_type = step["type"]
        cmd = Twist()

        # ---------------- 1. XỬ LÝ NGHỈ (PAUSE) ----------------
        if step_type == "PAUSE":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)

            elapsed = (self.get_clock().now() - self.pause_start_time).nanoseconds / 1e9
            if elapsed >= step["time"]:
                self.current_step_idx += 1
                self.init_step(self.current_step_idx)
            return

        # ---------------- 2. XỬ LÝ TIẾN / LÙI (FORWARD / BACKWARD) ----------------
        elif step_type in ["FORWARD", "BACKWARD"]:
            target_dist = step["dist"]
            # Tính khoảng cách đã di chuyển được từ điểm bắt đầu bước
            moved_dist = math.hypot(self.current_x - self.start_x, self.current_y - self.start_y)
            rem_dist = target_dist - moved_dist

            # Đã đến đích (sai số <= 0.015m)
            if rem_dist <= 0.015:
                cmd.linear.x = 0.0
                self.cmd_pub.publish(cmd)
                self.get_logger().info(f"✅ Hoàn thành {step['desc']} (Đã đi: {moved_dist:.2f}m)")
                self.current_step_idx += 1
                self.init_step(self.current_step_idx)
                return

            # P-Controller cho vận tốc dài
            v = self.kp_lin * rem_dist
            v = max(self.min_v, min(self.max_v, v))
            
            # Đổi chiều nếu là LÙI
            cmd.linear.x = v if step_type == "FORWARD" else -v
            self.cmd_pub.publish(cmd)

            self.get_logger().info(
                f"[{step['desc']}] Tiến độ: {moved_dist:.2f}m / {target_dist:.2f}m", 
                throttle_duration_sec=0.5
            )

        # ---------------- 3. XỬ LÝ QUAY (TURN) ----------------
        elif step_type == "TURN":
            target_deg = step["deg"]
            target_rad = math.radians(target_deg)
            rem_rad = abs(target_rad) - abs(self.accumulated_yaw)
            current_deg = math.degrees(abs(self.accumulated_yaw))

            # Đã quay đủ góc (sai số <= 0.5 độ)
            if rem_rad <= math.radians(0.5):
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                self.get_logger().info(f"✅ Hoàn thành {step['desc']} (Góc đạt: {current_deg:.2f}°)")
                self.current_step_idx += 1
                self.init_step(self.current_step_idx)
                return

            # P-Controller cho vận tốc góc
            w = self.kp_ang * rem_rad
            w = max(self.min_w, min(self.max_w, w))

            # Hướng quay: Dương = Trái, Âm = Phải
            cmd.angular.z = w if target_deg > 0 else -w
            self.cmd_pub.publish(cmd)

            self.get_logger().info(
                f"[{step['desc']}] Tiến độ: {current_deg:.1f}° / {abs(target_deg):.1f}°", 
                throttle_duration_sec=0.5
            )

def main(args=None):
    rclpy.init(args=args)
    node = SequenceMotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        node.cmd_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()