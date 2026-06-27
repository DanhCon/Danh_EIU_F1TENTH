#!/usr/bin/env python3
"""
MPC Controller tối ưu cho F1Tenth
Hợp nhất mpc_basic.py + tính năng tối ưu từ MAIN_MPC_car_general.py

Cải tiến từ MAIN_MPC:
  ① QP Solver có ràng buộc (qpsolvers + osqp)
  ② Ma trận ràng buộc G, ht tự sinh cho δ rate-limit + magnitude-limit
  ③ Tốc độ thích nghi theo độ cong đường đua (curvature-based speed)
  ④ Debug QP thất bại: in ma trận H, G, ht khi solver lỗi
  ⑤ Rate limit Δδ/bước: ngăn bẻ lái đột ngột phá lốp
  ⑥ Hai đầu vào: δ (lái) + U2 (gia tốc) với state-space mở rộng
"""

import rclpy
from rclpy.node import Node
import scipy.sparse as sparse
import math
import csv
import numpy as np
from copy import deepcopy

from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import Point

# ① Thử import QP solver – fallback về unconstrained nếu chưa cài
# Cài đặt: pip install qpsolvers[osqp] --break-system-packages
try:
    from qpsolvers import solve_qp
    QP_AVAILABLE = True
except ImportError:
    QP_AVAILABLE = False


def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class MPCNode(Node):
    def __init__(self):
        super().__init__("mpc_controller_node")

        # ============================================================
        # 1. THAM SỐ VẬT LÝ VÀ MPC CHO F1TENTH
        # ============================================================
        self.m   = 3.47
        self.Iz  = 0.04712
        self.Caf = 60.0
        self.Car = 60.0
        self.lf  = 0.158
        self.lr  = 0.171
        self.Ts  = 0.05       # Lấy mẫu 20 Hz

        self.hz      = 15     # Horizon 1 giây
        self.outputs = 2      # Đầu ra: (e_psi, e_y)
        self.inputs  = 2      # ⑥ Hai đầu vào: delta (δ) + gia tốc (a)

        # ── Ma trận trọng số ──────────────────────────────────────────
        # Q: phạt lỗi trạng thái dọc theo horizon
        self.Q = np.diag([10.0, 200.0])     # [e_psi, e_y]
        # S: phạt lỗi trạng thái cuối horizon (terminal cost)
        self.S = np.diag([10.0, 200.0])
        # R: phạt biến thiên đầu vào [Δδ, Δa]
        self.R = np.diag([ 8000.0, 50.0])     # R_a=50 cho phép thay đổi tốc độ mềm mại

        # ============================================================
        # 2. THAM SỐ RÀNG BUỘC (② ⑤) – từ MAIN_MPC
        # ============================================================
        # Ràng buộc góc lái tổng (magnitude)
        self.delta_max  =  0.35        # rad
        self.delta_min  = -0.35        # rad

        # ⑤ Rate limit: thay đổi góc lái tối đa mỗi bước Ts
        self.du_delta_max = 0.05       # rad/step (~1 rad/s ở 20 Hz)

        # ⑥ Ràng buộc gia tốc dọc
        self.a_max       =  3.0        # m/s²
        self.a_min       = -4.0        # m/s² (phanh mạnh hơn tăng tốc)
        self.du_a_max    =  1.0        # m/s³ (giật tối đa)

        # Giới hạn tốc độ
        self.v_max = 8.5               # m/s – tốc độ tối đa
        self.v_min = 0.8               # m/s – tốc độ tối thiểu

        # ③ Tham số tốc độ thích nghi theo độ cong
        self.curvature_lookahead = 13   # số waypoint để ước tính curvature
        self.v_straight = 7.0          # m/s trên đường thẳng
        self.v_curve    = 1.2          # m/s vào cua gắt

        # ============================================================
        # 3. KHỞI TẠO TRẠNG THÁI
        # ============================================================
        self.U1 = 0.0    # Góc lái hiện tại (rad)
        self.U2 = 0.0    # Gia tốc dọc hiện tại (m/s²)
        self.v_x_current = 1.0

        self.start_index = None
        self.waypoints   = []

        # ============================================================
        # 4. KHỞI TẠO ROS 2
        # ============================================================
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.car_frame   = "ego_racecar/base_link"
        self.map_frame   = "map"

        self.sub_odom        = self.create_subscription(
            Odometry, "ego_racecar/odom", self.odom_callback, 10)
        self.pub_drive       = self.create_publisher(
            AckermannDriveStamped, "/drive", 10)
        self.pub_marker_path = self.create_publisher(
            MarkerArray, "/publish_full_waypoint", 10)
        self.pub_mpc_ref     = self.create_publisher(
            Marker, "/mpc_lookahead_points", 10)
        self.pub_mpc_predict = self.create_publisher(Marker, "/mpc_predict_path", 10)

        csv_path = (
            "/sim_ws/install/waypoint/share/waypoint/"
            "f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        )
        self.load_waypoints(csv_path)
        self.publish_full_waypoint()

        self.last_mpc_time = self.get_clock().now()

        if QP_AVAILABLE:
            self.get_logger().info(
                "MPC tối ưu (QP + ràng buộc + 2 đầu vào) đã khởi động!")
        else:
            self.get_logger().warn(
                "qpsolvers chưa cài – chạy unconstrained. "
                "Cài bằng: pip install qpsolvers[osqp] --break-system-packages")

    # ================================================================
    # TÍNH STATE-SPACE (⑥ MỞ RỘNG 2 ĐẦU VÀO: δ + gia tốc)
    # Bilinear (Tustin) discretization để đảm bảo stability
    # ================================================================
    def calculate_state_space(self, v_x):
        """
        Trả về Ad, Bd, Cd cho mô hình xe đạp tuyến tính hóa tại v_x.

        State x = [v_y, e_psi, yaw_rate, e_y, v_x]  (5 states)
        Input u = [delta, accel]                      (2 inputs)
        Output y = [e_psi, e_y]                       (2 outputs)

        ⑥ v_x được thêm vào như state tích phân: v̇_x = a
           Cho phép MPC tối ưu hóa cả lái + tốc độ cùng lúc.
        """
        v_x = max(v_x, 1.0)

        A1 = -(2 * self.Caf + 2 * self.Car)                        / (self.m  * v_x)
        A2 = -v_x - (2 * self.Caf * self.lf - 2 * self.Car * self.lr) / (self.m  * v_x)
        A3 = -(2 * self.lf * self.Caf - 2 * self.lr * self.Car)   / (self.Iz * v_x)
        A4 = -(2 * self.lf**2 * self.Caf + 2 * self.lr**2 * self.Car) / (self.Iz * v_x)

        # Continuous-time A (5×5): thêm hàng/cột v_x (pure integrator)
        A_c = np.array([
            [A1, 0, A2, 0, 0],   # v_y
            [0,  0,  1, 0, 0],   # e_psi
            [A3, 0, A4, 0, 0],   # yaw_rate
            [1, v_x, 0, 0, 0],   # e_y
            [0,  0,  0, 0, 0],   # v_x (integrator)
        ])

        # Continuous-time B (5×2): [B_delta | B_accel]
        B_c = np.array([
            [2 * self.Caf / self.m,         0],
            [0,                             0],
            [2 * self.lf * self.Caf / self.Iz, 0],
            [0,                             0],
            [0,                             1],   # a → v̇_x = a
        ])

        # Output matrix C (2×5): lấy e_psi (index 1) và e_y (index 3)
        C_c = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ])

        # Bilinear (Tustin) discretization – đảm bảo stability
        I       = np.eye(5)
        inv_term = np.linalg.inv(I - (self.Ts / 2.0) * A_c)
        Ad = inv_term @ (I + (self.Ts / 2.0) * A_c)
        Bd = inv_term @ (B_c * self.Ts)
        Cd = C_c

        return Ad, Bd, Cd

    # ================================================================
    # MPC SIMPLIFICATION (⑥ cập nhật cho 2 đầu vào)
    # ================================================================
    def mpc_simplification(self, Ad, Bd, Cd, hz):
        """
        Xây dựng các ma trận tối giản cho bài toán QP.

        Ad: (5,5),  Bd: (5,2),  Cd: (2,5)
        Augmented state: x_aug = [x (5), U1, U2]  → kích thước 7
        Augmented A_aug: (7,7),  B_aug: (7,2),  C_aug: (2,7)

        Trả về: Hdb, Fdbt, Cdb, Adc
        """
        n_x   = Ad.shape[0]   # 5
        n_u   = Bd.shape[1]   # 2
        n_y   = Cd.shape[0]   # 2
        n_aug = n_x + n_u     # 7

        # --- Xây augmented matrices ---
        A_aug = np.block([
            [Ad,                    Bd                  ],
            [np.zeros((n_u, n_x)), np.eye(n_u)         ],
        ])   # (7,7)

        B_aug = np.block([
            [Bd               ],
            [np.eye(n_u)      ],
        ])   # (7,2)

        C_aug = np.block([
            [Cd, np.zeros((n_y, n_u))],
        ])   # (2,7)

        CQC = C_aug.T @ self.Q @ C_aug   # (7,7)
        CSC = C_aug.T @ self.S @ C_aug   # (7,7)
        QC  = self.Q @ C_aug             # (2,7)
        SC  = self.S @ C_aug             # (2,7)

        # --- Kích thước block ---
        s_x = n_aug * hz          # state block rows  = 7*hz
        s_y = n_y   * hz          # output block rows = 2*hz
        s_u = n_u   * hz          # input block rows  = 2*hz

        Qdb = np.zeros((s_x, s_x))
        Tdb = np.zeros((s_y, s_x))
        Rdb = np.zeros((s_u, s_u))
        Cdb = np.zeros((s_x, s_u))
        Adc = np.zeros((s_x, n_aug))

        for i in range(hz):
            # Trọng số cuối horizon dùng S (terminal cost)
            if i == hz - 1:
                Qdb[n_aug*i:n_aug*(i+1), n_aug*i:n_aug*(i+1)] = CSC
                Tdb[n_y*i:n_y*(i+1),   n_aug*i:n_aug*(i+1)] = SC
            else:
                Qdb[n_aug*i:n_aug*(i+1), n_aug*i:n_aug*(i+1)] = CQC
                Tdb[n_y*i:n_y*(i+1),   n_aug*i:n_aug*(i+1)] = QC

            Rdb[n_u*i:n_u*(i+1), n_u*i:n_u*(i+1)] = self.R

            for j in range(hz):
                if j <= i:
                    Cdb[n_aug*i:n_aug*(i+1), n_u*j:n_u*(j+1)] = (
                        np.linalg.matrix_power(A_aug, i - j) @ B_aug
                    )

            Adc[n_aug*i:n_aug*(i+1), :] = np.linalg.matrix_power(A_aug, i + 1)

        Hdb  = Cdb.T @ Qdb @ Cdb + Rdb      # (2*hz, 2*hz)  – ma trận Hessian QP
        temp1 = Adc.T @ Qdb @ Cdb            # (n_aug, 2*hz)
        temp2 = -Tdb @ Cdb                   # (2*hz, 2*hz)
        Fdbt  = np.vstack((temp1, temp2))    # (n_aug + 2*hz, 2*hz)

        return Hdb, Fdbt, Cdb, Adc

    # ================================================================
    # ① ② ⑤  XÂY MA TRẬN RÀNG BUỘC G, ht  (từ MAIN_MPC)
    # ================================================================
    def build_constraints(self, hz):
        """
        Tạo ma trận bất đẳng thức: G * du <= ht
        du = [dδ_0, da_0, dδ_1, da_1, ..., dδ_{hz-1}, da_{hz-1}]  kích thước 2*hz

        Ràng buộc:
          ⑤ Rate limit mỗi bước: |dδ_k| <= du_delta_max, |da_k| <= du_a_max
          ② Magnitude tổng:       delta_min <= U1 + Σdδ <= delta_max
                                   a_min    <= U2 + Σda  <= a_max
        """
        n = hz          # số bước
        I2 = np.eye(2)  # block 2×2 cho [dδ, da]
        n2 = 2 * n      # tổng số biến quyết định

        # ── ⑤ Rate limit: ±I * du <= rate_max ─────────────────────
        I_block = np.zeros((n2, n2))
        for k in range(n):
            I_block[2*k:2*k+2, 2*k:2*k+2] = I2

        rate_max = np.tile([self.du_delta_max, self.du_a_max], n)

        G_rate_pos = I_block                   # du  <= rate_max
        G_rate_neg = -I_block                  # -du <= rate_max

        # ── ② Magnitude: cumulative sum matrix (lower-triangular) ──
        # L[i,j] = I2 nếu j <= i, ngược lại 0
        L = np.zeros((n2, n2))
        for i in range(n):
            for j in range(n):
                if j <= i:
                    L[2*i:2*i+2, 2*j:2*j+2] = I2

        # U + L*du <= U_max  →  L*du <= U_max - U_current
        # -(U + L*du) <= -U_min  →  -L*du <= U_current - U_min
        U_current = np.tile([self.U1, self.U2], n)
        U_max_vec = np.tile([self.delta_max, self.a_max],  n)
        U_min_vec = np.tile([self.delta_min, self.a_min],  n)

        G_mag_pos =  L
        G_mag_neg = -L

        ht_rate_pos = rate_max
        ht_rate_neg = rate_max
        ht_mag_pos  = U_max_vec - U_current
        ht_mag_neg  = U_current - U_min_vec

        G  = np.vstack([G_rate_pos, G_rate_neg, G_mag_pos, G_mag_neg])
        ht = np.concatenate([ht_rate_pos, ht_rate_neg, ht_mag_pos, ht_mag_neg])

        return G.astype(np.float64), ht.astype(np.float64)

    # ================================================================
    # ③ TỐC ĐỘ THÍCH NGHI THEO ĐỘ CONG (từ MAIN_MPC)
    # ================================================================
    def compute_target_speed(self, nearest_idx):
        """
        Tính tốc độ mục tiêu dựa trên độ cong đường đua tại vị trí hiện tại.
        Đường thẳng → v_straight, cua gắt → v_curve.
        """
        n = len(self.waypoints)
        k = self.curvature_lookahead

        # Lấy 3 điểm: trước, hiện tại, sau
        p_prev = self.waypoints[(nearest_idx - k) % n]
        p_curr = self.waypoints[nearest_idx]
        p_next = self.waypoints[(nearest_idx + k) % n]

        # Vector tiếp tuyến
        dx1 = p_curr[0] - p_prev[0]
        dy1 = p_curr[1] - p_prev[1]
        dx2 = p_next[0] - p_curr[0]
        dy2 = p_next[1] - p_curr[1]

        # Độ cong xấp xỉ: cross product / (norm1 * norm2)
        cross  = abs(dx1 * dy2 - dy1 * dx2)
        norm1  = math.hypot(dx1, dy1) + 1e-6
        norm2  = math.hypot(dx2, dy2) + 1e-6
        curvature = cross / (norm1 * norm2)   # ~ sin(angle_change)

        # Ánh xạ tuyến tính: curvature ∈ [0, 1] → speed ∈ [v_straight, v_curve]
        curvature_clipped = min(curvature, 1.0)
        target_speed = self.v_straight + curvature_clipped * (self.v_curve - self.v_straight)
        target_speed = max(self.v_min, min(self.v_max, target_speed))

        return target_speed

    # ================================================================
    # WAYPOINT LOADING & PATH SMOOTHING (giữ nguyên từ mpc_basic)
    # ================================================================
    def load_waypoints(self, filename):
        raw_waypoints = []
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    line_data = row[0].split() if len(row) == 1 else row
                    try:
                        raw_waypoints.append([float(line_data[0]), float(line_data[1])])
                    except ValueError:
                        continue

            smoothed = self.smooth_path(raw_waypoints)
            self.waypoints = []
            for i in range(len(smoothed)):
                p1  = smoothed[i]
                p2  = smoothed[(i + 1) % len(smoothed)]
                yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                self.waypoints.append([p1[0], p1[1], yaw])

        except Exception as e:
            self.get_logger().error(f"LỖI ĐỌC FILE: {e}")

    def smooth_path(self, path, weight_data=0.5, weight_smooth=0.2, tolerance=0.00001):
        new_path = deepcopy(path)
        change   = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path) - 1):
                ax, ay = new_path[i][0], new_path[i][1]
                new_path[i][0] += (weight_data   * (path[i][0] - new_path[i][0])
                                 + weight_smooth * (new_path[i-1][0] + new_path[i+1][0]
                                                    - 2.0 * new_path[i][0]))
                new_path[i][1] += (weight_data   * (path[i][1] - new_path[i][1])
                                 + weight_smooth * (new_path[i-1][1] + new_path[i+1][1]
                                                    - 2.0 * new_path[i][1]))
                change += abs(ax - new_path[i][0]) + abs(ay - new_path[i][1])
        return new_path

    # ================================================================
    # CALLBACK CHÍNH
    # ================================================================
    def odom_callback(self, msg: Odometry):
        if not self.waypoints:
            return

        # Chặn tần số 20 Hz
        current_time = self.get_clock().now()
        if (current_time - self.last_mpc_time).nanoseconds / 1e9 < self.Ts:
            return
        self.last_mpc_time = current_time

        v_x      = msg.twist.twist.linear.x
        v_y      = msg.twist.twist.linear.y
        yaw_rate = msg.twist.twist.angular.z
        self.v_x_current = max(v_x, self.v_min)

        # Lấy pose từ TF
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.car_frame, rclpy.time.Time(seconds=0))
            rx   = transform.transform.translation.x
            ry   = transform.transform.translation.y
            q    = transform.transform.rotation
            r_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        except TransformException:
            return

        # Tìm waypoint gần nhất (rolling window)
        nearest_idx = self._find_nearest_waypoint(rx, ry)

        # Tính lỗi cục bộ
        wp_x, wp_y, wp_yaw = self.waypoints[nearest_idx]
        e_psi = normalize_angle(r_yaw - wp_yaw)
        dx    = rx - wp_x
        dy    = ry - wp_y
        e_y   = -math.sin(wp_yaw) * dx + math.cos(wp_yaw) * dy

        # ③ Tốc độ mục tiêu thích nghi
        v_target = self.compute_target_speed(nearest_idx)
        v_error  = v_target - self.v_x_current   # dùng làm tham chiếu cho e_vx

        # Xây state-space (⑥ state bổ sung v_x)
        Ad, Bd, Cd = self.calculate_state_space(self.v_x_current)
        Hdb, Fdbt, Cdb, Adc = self.mpc_simplification(Ad, Bd, Cd, self.hz)

        # Augmented state: [v_y, e_psi, yaw_rate, e_y, v_x, U1, U2]
        states     = np.array([v_y, e_psi, yaw_rate, e_y, self.v_x_current])
        x_aug_t    = np.concatenate((states, [self.U1, self.U2]))  # (7,)

        # Xây reference trajectory (nhìn trước hz bước)
        r_vector, ref_global_points = self._build_reference(
            nearest_idx, wp_x, wp_y, wp_yaw, v_target)

        self.publish_mpc_reference(ref_global_points)

        # ──────────────────────────────────────────────────────────
        # GIẢI QP (① ② ⑤ từ MAIN_MPC)
        # Minimize: 0.5 * du^T * H * du + ft^T * du
        # Subject to: G * du <= ht
        # ──────────────────────────────────────────────────────────
        ft_input = np.concatenate((x_aug_t, r_vector))  # (7 + 2*hz,)
        ft       = (Fdbt.T @ ft_input).astype(np.float64)
        Hdb_f64  = Hdb.astype(np.float64)

        # Đảm bảo Hdb symmetry & PSD (tránh numerical drift)
        Hdb_f64 = 0.5 * (Hdb_f64 + Hdb_f64.T)
        Hdb_f64 += 1e-8 * np.eye(Hdb_f64.shape[0])

        du = self._solve(Hdb_f64, ft)
        if du is None:
            return

        # Cập nhật đầu vào
        self.U1 = float(np.clip(self.U1 + du[0], self.delta_min, self.delta_max))
        self.U2 = float(np.clip(self.U2 + du[1], self.a_min,     self.a_max    ))
        # ──────────────────────────────────────────────────────────
        # TÍNH TOÁN ĐƯỜNG ĐI DỰ ĐOÁN (PREDICTED TRAJECTORY)
        # ──────────────────────────────────────────────────────────
        # Vector X_pred kích thước (7 * hz, 1) chứa toàn bộ state tương lai
        X_pred = Adc @ x_aug_t + Cdb @ du
        predict_global_points = []
        s_accum = 0.0  # Khoảng cách di chuyển dọc theo tiếp tuyến

        for i in range(self.hz):
            # Trích xuất state thứ i (mỗi state có 7 phần tử)
            # Index 3 là e_y (sai số ngang), Index 4 là v_x (vận tốc dọc)
            e_y_pred = X_pred[7 * i + 3]
            v_x_pred = X_pred[7 * i + 4]

            # Tính quãng đường dọc đã đi được trong i bước
            s_accum += v_x_pred * self.Ts

            # Phép biến đổi Affine (Xoay + Tịnh tiến) từ Local sang Global Frame
            px = wp_x + s_accum * math.cos(wp_yaw) - e_y_pred * math.sin(wp_yaw)
            py = wp_y + s_accum * math.sin(wp_yaw) + e_y_pred * math.cos(wp_yaw)

            predict_global_points.append((px, py))

        # Publish đường dự đoán
        self.publish_mpc_predict(predict_global_points, rx, ry)
        # ──────────────────────────────────────────────────────────

        # Tích phân tốc độ đơn giản (giới hạn v_x trong [v_min, v_max])
        v_x_next = np.clip(self.v_x_current + self.U2 * self.Ts,
                           self.v_min, self.v_max)
        self.v_x_current = float(v_x_next)

        # Xuất lệnh lái
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = self.U1
        drive_msg.drive.speed          = float(v_target)   # ③ tốc độ thích nghi
        self.pub_drive.publish(drive_msg)
        self.get_logger().info(f"toc do {drive_msg.drive.speed}")

    # ================================================================
    # ① SOLVE: QP có ràng buộc (MAIN_MPC) hoặc fallback unconstrained
    # ================================================================
    def publish_mpc_predict(self, points_list, current_x, current_y):
        """Vẽ đường đi dự đoán của MPC (Predicted Trajectory) lên RViz2"""
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "mpc_predict"
        marker.id = 1
        
        # LINE_STRIP nối các điểm lại thành một đường liền mạch
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.08  # Độ dày đường (8cm)
        
        # Màu Đỏ (Red) đặc trưng cho Prediction
        marker.color.a = 0.8
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        
        # Điểm bắt đầu chính là vị trí HIỆN TẠI của xe (nối đuôi cho mượt)
        p_start = Point()
        p_start.x = float(current_x)
        p_start.y = float(current_y)
        p_start.z = 0.15 # Nổi lên trên track một chút
        marker.points.append(p_start)

        # Nối các điểm tương lai do QP Solver tính toán
        for pt in points_list:
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = 0.15
            marker.points.append(p)
            
        self.pub_mpc_predict.publish(marker)
    def _solve(self, Hdb, ft):
        """
        ① Giải QP bằng solve_qp (osqp) với ràng buộc G, ht.
        ④ Debug: in thông tin chi tiết khi solver thất bại (từ MAIN_MPC).
        Nếu qpsolvers chưa cài, fallback về inverse (không ràng buộc).
        """
        if QP_AVAILABLE:
            G, ht = self.build_constraints(self.hz)
            Hdb_f64 = Hdb.astype(np.float64)
            Hdb_f64 = 0.5 * (Hdb_f64 + Hdb_f64.T) + 1e-8 * np.eye(Hdb_f64.shape[0])
            
            Hdb_sparse = sparse.csc_matrix(Hdb_f64)
            G_sparse   = sparse.csc_matrix(G)
            try:
                du = solve_qp(
                    Hdb_sparse, ft,
                    G=G_sparse, h=ht,
                    solver="osqp",
                    verbose=False,
                    eps_abs=1e-5, eps_rel=1e-5,
                )
                if du is None:
                    raise ValueError("solve_qp trả về None")
                return du  # shape (2*hz,)

            except (ValueError, Exception) as e:
                # ④ Debug QP thất bại (từ MAIN_MPC)
                self.get_logger().warn(f"[QP thất bại] {e}")
                self.get_logger().warn(
                    f"  Hdb cond = {np.linalg.cond(Hdb):.2e} | "
                    f"ft norm = {np.linalg.norm(ft):.4f} | "
                    f"U1={self.U1:.3f} U2={self.U2:.3f}"
                )
                # Fallback về unconstrained trong bước lỗi
                return self._solve_unconstrained(Hdb, ft)
        else:
            return self._solve_unconstrained(Hdb, ft)

    def _solve_unconstrained(self, Hdb, ft):
        """Giải unconstrained QP: du = -H^{-1} * ft"""
        try:
            du = -np.linalg.solve(Hdb, ft)  # dùng solve thay inv để ổn định hơn
            return du
        except np.linalg.LinAlgError as e:
            self.get_logger().error(f"[LinAlgError] {e}")
            return None

    # ================================================================
    # HELPER: TÌM WAYPOINT GẦN NHẤT
    # ================================================================
    def _find_nearest_waypoint(self, rx, ry):
        if self.start_index is None:
            min_dist = float("inf")
            idx = 0
            for i, p in enumerate(self.waypoints):
                d = math.hypot(rx - p[0], ry - p[1])
                if d < min_dist:
                    min_dist = d
                    idx = i
            self.start_index = idx
            return idx

        idx      = self.start_index
        curr_d   = math.hypot(rx - self.waypoints[idx][0], ry - self.waypoints[idx][1])
        for _ in range(20):
            nxt   = (idx + 1) % len(self.waypoints)
            nxt_d = math.hypot(rx - self.waypoints[nxt][0], ry - self.waypoints[nxt][1])
            if nxt_d < curr_d:
                idx    = nxt
                curr_d = nxt_d
            else:
                break
        self.start_index = idx
        return idx

    # ================================================================
    # HELPER: XÂY REFERENCE TRAJECTORY (giữ nguyên + thêm v_x ref)
    # ================================================================
    def _build_reference(self, nearest_idx, wp_x, wp_y, wp_yaw, v_target):
        """
        Tính reference vector r_vector (2*hz,) cho [e_psi_ref, e_y_ref].
        Cũng trả về danh sách điểm toàn cục để vẽ trên RViz.
        """
        r_list            = []
        ref_global_points = []
        step_dist = max(self.v_x_current, 2.5) * self.Ts
        curr_idx  = nearest_idx
        dist_accum = 0.0

        for i in range(1, self.hz + 1):
            target_dist = i * step_dist

            # Nội suy liên tục dọc waypoint
            while True:
                next_idx    = (curr_idx + 1) % len(self.waypoints)
                p1          = self.waypoints[curr_idx]
                p2          = self.waypoints[next_idx]
                segment_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

                if dist_accum + segment_len >= target_dist:
                    ratio  = ((target_dist - dist_accum) / segment_len
                              if segment_len > 0 else 0.0)
                    fx     = p1[0] + ratio * (p2[0] - p1[0])
                    fy     = p1[1] + ratio * (p2[1] - p1[1])
                    fyaw   = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    break
                else:
                    dist_accum += segment_len
                    curr_idx    = next_idx

            ref_global_points.append((fx, fy))

            # Lỗi tương lai so với gốc cục bộ
            fdx     = fx - wp_x
            fdy     = fy - wp_y
            local_y   = -math.sin(wp_yaw) * fdx + math.cos(wp_yaw) * fdy
            local_yaw = normalize_angle(fyaw - wp_yaw)

            r_list.extend([local_yaw, local_y])

        return np.array(r_list), ref_global_points

    # ================================================================
    # PUBLISH MARKERS (giữ nguyên từ mpc_basic)
    # ================================================================
    def publish_mpc_reference(self, points_list):
        marker             = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.ns              = "mpc_reference"
        marker.id              = 0
        marker.type            = Marker.SPHERE_LIST
        marker.action          = Marker.ADD
        marker.scale.x         = 0.15
        marker.scale.y         = 0.15
        marker.scale.z         = 0.15
        marker.color.a         = 1.0
        marker.color.r         = 0.0
        marker.color.g         = 1.0
        marker.color.b         = 1.0
        for pt in points_list:
            p   = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = 0.1
            marker.points.append(p)
        self.pub_mpc_ref.publish(marker)

    def publish_full_waypoint(self):
        marker_array           = MarkerArray()
        marker                 = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.id              = 0
        marker.type            = Marker.LINE_STRIP
        marker.action          = Marker.ADD
        marker.scale.x         = 0.05
        marker.color.a         = 1.0
        marker.color.r         = 1.0
        marker.color.g         = 1.0
        marker.color.b         = 0.0
        for point in self.waypoints:
            p   = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.0
            marker.points.append(p)
        if self.waypoints:
            p   = Point()
            p.x = float(self.waypoints[0][0])
            p.y = float(self.waypoints[0][1])
            p.z = 0.0
            marker.points.append(p)
        marker_array.markers.append(marker)
        self.pub_marker_path.publish(marker_array)


# ====================================================================
# MAIN
# ====================================================================
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MPCNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[MPC ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            # Dừng xe an toàn trước khi tắt
            stop_msg = AckermannDriveStamped()
            stop_msg.drive.speed          = 0.0
            stop_msg.drive.steering_angle = 0.0
            node.pub_drive.publish(stop_msg)
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()