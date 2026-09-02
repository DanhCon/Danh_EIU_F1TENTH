
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import scipy.linalg as la
import math
import json
import os

# ================= 1. KHỐI BỘ LỌC EKF =================
class EKFEstimator:
    def __init__(self):
        self.x = np.zeros((3, 1))
        self.P = np.eye(3) * 0.05
        self.Q = np.diag([0.005, 0.005, 0.01])
        self.R = np.diag([0.02, 0.02, 0.03])
        self.is_initialized = False

    def update(self, x_meas, y_meas, theta_meas, dt):
        if not self.is_initialized:
            self.x = np.array([[x_meas], [y_meas], [theta_meas]])
            self.is_initialized = True
            return x_meas, y_meas, theta_meas

        theta = self.x[2, 0]

        # Predict
        F = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        P_pred = F @ self.P @ F.T + self.Q

        # Update
        z_meas = np.array([[x_meas], [y_meas], [theta_meas]])
        y_res = z_meas - self.x
        y_res[2, 0] = (y_res[2, 0] + np.pi) % (2 * np.pi) - np.pi

        H = np.eye(3)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_res
        self.P = (np.eye(3) - K @ H) @ P_pred
        self.x[2, 0] = (self.x[2, 0] + np.pi) % (2 * np.pi) - np.pi

        return self.x[0, 0], self.x[1, 0], self.x[2, 0]

# ================= 2. KHỐI BỘ ĐIỀU KHIỂN LQR =================
class LQRController:
    def __init__(self, dt=0.05):
        self.dt = dt
        # Ma trận trọng số LQR (Có thể tinh chỉnh nếu muốn bám gắt hơn)
        self.Q = np.diag([3.0, 25.0, 3.0])  # q_x, q_y (ép bám sát lề), q_theta
        self.R = np.diag([0.2, 0.2])        # r_v, r_w (mượt tay lái)

        self.max_v = 1.0   # m/s
        self.max_w = 0.8   # rad/s

    def compute_control(self, current_pose, target_pose, v_d, w_d):
        x, y, theta = current_pose
        x_d, y_d, theta_d = target_pose

        # 1. Sai số trong Robot Local Frame
        dx = x_d - x
        dy = y_d - y
        e_theta = (theta_d - theta + np.pi) % (2 * np.pi) - np.pi

        e_x = math.cos(theta) * dx + math.sin(theta) * dy
        e_y = -math.sin(theta) * dx + math.cos(theta) * dy
        e = np.array([[e_x], [e_y], [e_theta]])

        # 2. Tuyến tính hóa ma trận A, B
        A = np.array([
            [0.0, w_d, 0.0],
            [-w_d, 0.0, v_d],
            [0.0, 0.0, 0.0]
        ])
        B = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0]
        ])

        A_d = np.eye(3) + A * self.dt
        B_d = B * self.dt

        # 3. Giải Riccati tính Gain K
        try:
            P = la.solve_discrete_are(A_d, B_d, self.Q, self.R)
            K = la.inv(self.R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
        except Exception:
            K = np.zeros((2, 3))

        # 4. Đầu ra vận tốc [v_cmd, w_cmd]
        u = -K @ e
        v_cmd = v_d * math.cos(e_theta) + u[0, 0]
        w_cmd = w_d + u[1, 0]

        v_cmd = np.clip(v_cmd, -self.max_v, self.max_v)
        w_cmd = np.clip(w_cmd, -self.max_w, self.max_w)

        return v_cmd, w_cmd, e_x, e_y

# ================= 3. ROS 2 NODE CHÍNH =================
class LQRTrajectoryFollowerNode(Node):
    def __init__(self):
        super().__init__('lqr_trajectory_follower_node')

        self.declare_parameter('filename', 'trajectory.json')
        self.declare_parameter('lookahead_points', 2) # Nhìn trước 2 điểm để cua mượt

        self.filename = self.get_parameter('filename').value
        self.lookahead = self.get_parameter('lookahead_points').value

        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.ekf = EKFEstimator()
        self.lqr = LQRController(dt=0.05)

        self.waypoints = []
        self.load_trajectory()

        self.current_idx = 0
        self.odom_received = False
        self.raw_x = 0.0
        self.raw_y = 0.0
        self.raw_yaw = 0.0

        # Timer 20Hz
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("=== READY: NODE BÁM QUỸ ĐẠO EKF + LQR ĐÃ SẴN SÀNG ===")

    def load_trajectory(self):
        cwd = os.getcwd()
        path = os.path.join(cwd, self.filename)
        if not os.path.exists(path):
            self.get_logger().error(f"❌ KHÔNG TÌM THẤY FILE QUỸ ĐẠO TẠI: {path}")
            return

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.waypoints = data["waypoints"]
        self.get_logger().info(f"✅ Đã nạp file quỹ đạo thành công với {len(self.waypoints)} điểm Waypoint.")

    def get_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        self.odom_received = True
        self.raw_x = msg.pose.pose.position.x
        self.raw_y = msg.pose.pose.position.y
        self.raw_yaw = self.get_yaw(msg.pose.pose.orientation)

    def find_nearest_waypoint(self, est_x, est_y):
        """ Tìm điểm gần vị trí hiện tại của xe nhất """
        min_d = float('inf')
        best_i = self.current_idx
        
        search_start = max(0, self.current_idx - 5)
        search_end = min(len(self.waypoints), self.current_idx + 15)

        for i in range(search_start, search_end):
            wpt = self.waypoints[i]
            d = math.hypot(wpt['x'] - est_x, wpt['y'] - est_y)
            if d < min_d:
                min_d = d
                best_i = i
        return best_i

    def control_loop(self):
        if not self.odom_received or not self.waypoints:
            return

        # 1. Cập nhật vị trí mịn từ EKF
        est_x, est_y, est_yaw = self.ekf.update(self.raw_x, self.raw_y, self.raw_yaw, dt=0.05)

        # 2. Tìm điểm gần nhất & Điểm nhìn trước (Lookahead)
        self.current_idx = self.find_nearest_waypoint(est_x, est_y)
        target_idx = min(len(self.waypoints) - 1, self.current_idx + self.lookahead)
        
        target_wpt = self.waypoints[target_idx]
        target_pose = [target_wpt['x'], target_wpt['y'], target_wpt['yaw_rad']]
        
        # Lấy tốc độ mong muốn từ quỹ đạo (mặc định nếu dừng thì cho v_d = 0.25 m/s)
        v_d = max(0.15, abs(target_wpt['v_linear']))
        w_d = target_wpt['w_angular']

        # 3. Tính toán LQR
        cmd = Twist()
        dist_to_final = math.hypot(self.waypoints[-1]['x'] - est_x, self.waypoints[-1]['y'] - est_y)

        # KIỂM TRA ĐÃ ĐẾN ĐÍCH CUỐI CÙNG CHƯA
        if self.current_idx >= len(self.waypoints) - 2 or (dist_to_final < 0.08 and self.current_idx > len(self.waypoints) * 0.8):
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.get_logger().info("🎉 ✅ ĐÃ BÁM HOÀN THÀNH HẾT QUỸ ĐẠO! DỪNG XE.", throttle_duration_sec=2.0)
            return

        # Tính toán lệnh tốc độ LQR
        v_cmd, w_cmd, e_x, e_y = self.lqr.compute_control(
            current_pose=[est_x, est_y, est_yaw],
            target_pose=target_pose,
            v_d=v_d, w_d=w_d
        )

        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"🚀 [Bám Quỹ Đạo #{self.current_idx}/{len(self.waypoints)}] Sai số lệch lề (e_y): {e_y*100:.1f}cm | V_cmd: {v_cmd:.2f}m/s", 
            throttle_duration_sec=0.4
        )

def main(args=None):
    rclpy.init(args=args)
    node = LQRTrajectoryFollowerNode()
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