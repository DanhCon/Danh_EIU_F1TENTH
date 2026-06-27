#!/usr/bin/env python3
"""
MPC Controller + LiDAR Obstacle Avoidance (3-Layer) cho F1Tenth
================================================================
Tích hợp từ: mpc_basic.py + MAIN_MPC_car_general.py + LiDAR tránh vật cản

Kiến trúc 3 lớp tránh vật cản:
  Lớp 1 (Soft)       – Trọng số Q động: tăng phạt e_y theo phía vật cản
  Lớp 2 (Hard)       – Siết ràng buộc δ: thu hẹp biên góc lái về phía nguy hiểm
  Lớp 3 (Proactive)  – Dịch reference trajectory: MPC chủ động "muốn" tránh
  Emergency          – Phanh khẩn nếu d_front < d_stop

Cài đặt:
  pip install qpsolvers[osqp] --break-system-packages
"""

import rclpy
from rclpy.node import Node
import math
import csv
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import Point

try:
    from qpsolvers import solve_qp
    QP_AVAILABLE = True
except ImportError:
    QP_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────────────
# TIỆN ÍCH
# ────────────────────────────────────────────────────────────────────────────

def euler_from_quaternion(x, y, z, w) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def normalize_angle(angle: float) -> float:
    while angle >  math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle


# ────────────────────────────────────────────────────────────────────────────
# CẤU TRÚC DỮ LIỆU LIDAR
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ObstacleInfo:
    """Kết quả phân tích LiDAR theo 5 sector (góc tính từ mũi xe, dương = trái)."""
    d_front:       float = 999.0   # sector [-10°, +10°]   thẳng trước
    d_front_left:  float = 999.0   # sector [+10°, +60°]   trước-trái
    d_front_right: float = 999.0   # sector [-60°, -10°]   trước-phải
    d_left:        float = 999.0   # sector [+60°, +120°]  thuần trái
    d_right:       float = 999.0   # sector [-120°, -60°]  thuần phải

    # Kết quả tính toán từ 3 lớp (được fill bởi compute_*())
    q_weight_left:    float = 0.0  # Lớp 1: tăng thêm cho Q_y phía trái
    q_weight_right:   float = 0.0  # Lớp 1: tăng thêm cho Q_y phía phải
    delta_max_eff:    float = 0.4  # Lớp 2: biên lái tối đa hiệu dụng
    delta_min_eff:    float = -0.4 # Lớp 2: biên lái tối thiểu hiệu dụng
    ref_lateral_offset: float = 0.0 # Lớp 3: dịch ngang reference [m]
    emergency_brake:  bool  = False # Phanh khẩn
    v_limit_obs:      float = 999.0 # Giới hạn tốc độ do vật cản phía trước


# ────────────────────────────────────────────────────────────────────────────
# NODE MPC + LIDAR
# ────────────────────────────────────────────────────────────────────────────

class MPCNode(Node):

    def __init__(self):
        super().__init__("mpc_lidar_node")

        # ── Tham số vật lý F1Tenth ────────────────────────────────────────
        self.m   = 3.47
        self.Iz  = 0.04712
        self.Caf = 60.0
        self.Car = 60.0
        self.lf  = 0.158
        self.lr  = 0.171
        self.Ts  = 0.05         # 20 Hz

        self.hz      = 13       # Horizon
        self.outputs = 2        # [e_psi, e_y]
        self.inputs  = 2        # [δ, a]

        # ── Trọng số MPC cơ bản ───────────────────────────────────────────
        self.Q_base = np.diag([10.0, 100.0])   # [e_psi, e_y] baseline
        self.S_base = np.diag([10.0, 100.0])
        self.R      = np.diag([8000.0, 50.0])

        # ── Ràng buộc điều khiển ─────────────────────────────────────────
        self.delta_max   =  0.40
        self.delta_min   = -0.40
        self.du_delta_max = 0.05   # rad/step
        self.a_max       =  3.0
        self.a_min       = -4.0
        self.du_a_max    =  1.0

        # ── Tốc độ ───────────────────────────────────────────────────────
        self.v_max         = 3.5
        self.v_min         = 0.8
        self.v_straight    = 3.5
        self.v_curve       = 1.2
        self.curvature_lookahead = 5

        # ── Tham số LiDAR tránh vật cản ──────────────────────────────────
        # Khoảng cách an toàn: bắt đầu can thiệp
        self.d_safe_front  = 2.0   # m – phía trước
        self.d_safe_side   = 1.0   # m – hai bên
        # Khoảng cách nguy hiểm: can thiệp mạnh
        self.d_danger      = 0.8   # m
        # Khoảng cách dừng khẩn
        self.d_stop        = 0.35  # m

        # Lớp 1: Mức tăng Q_y tối đa khi có vật cản cạnh sườn
        self.q_obs_max     = 800.0  # ≈ 8× Q_y_base – đủ mạnh để "sợ"
        # Lớp 2: Tỉ lệ thu hẹp biên lái tối thiểu (giữ 20% biên khi sát vật)
        self.constraint_shrink_min = 0.20
        # Lớp 3: Offset ngang tối đa khi vật cản gần nhất
        self.ref_offset_max = 0.25  # m – không quá 25cm tránh ra ngoài track

        # ── Trạng thái ────────────────────────────────────────────────────
        self.U1            = 0.0
        self.U2            = 0.0
        self.v_x_current   = 1.0
        self.start_index   = None
        self.waypoints     = []
        self.obstacle_info = ObstacleInfo()  # Kết quả LiDAR mới nhất

        # ── ROS 2 ─────────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.car_frame   = "ego_racecar/base_link"
        self.map_frame   = "map"

        self.sub_odom  = self.create_subscription(
            Odometry, "ego_racecar/odom", self.odom_callback, 10)
        self.sub_scan  = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10)
        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, "/drive", 10)
        self.pub_marker_path = self.create_publisher(
            MarkerArray, "/publish_full_waypoint", 10)
        self.pub_mpc_ref = self.create_publisher(
            Marker, "/mpc_lookahead_points", 10)
        self.pub_obs_marker = self.create_publisher(
            Marker, "/obstacle_zone", 10)

        csv_path = (
            "/sim_ws/install/waypoint/share/waypoint/"
            "f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        )
        self.load_waypoints(csv_path)
        self.publish_full_waypoint()

        self.last_mpc_time = self.get_clock().now()

        mode = "QP+ràng buộc" if QP_AVAILABLE else "unconstrained (cài qpsolvers!)"
        self.get_logger().info(
            f"MPC+LiDAR (3-lớp tránh vật cản, {mode}) đã khởi động!")

    # ════════════════════════════════════════════════════════════════════════
    # CALLBACK LIDAR – xử lý mỗi scan mới
    # ════════════════════════════════════════════════════════════════════════

    def scan_callback(self, msg: LaserScan):
        """
        Parse LaserScan thành 5 sector và tính toán 3 lớp can thiệp.
        Chạy ở tần số riêng, kết quả lưu vào self.obstacle_info.
        """
        obs = self._parse_sectors(msg)
        self._layer1_weight_scaling(obs)
        self._layer2_constraint_tightening(obs)
        self._layer3_ref_offset(obs)
        self._compute_speed_limit(obs)
        self.obstacle_info = obs
        self._publish_obstacle_marker(obs)

    def _parse_sectors(self, msg: LaserScan) -> ObstacleInfo:
        """
        Chia scan 270° thành 5 sector, lấy khoảng cách nhỏ nhất trong mỗi sector.

        Quy ước góc: angle_min → angle_max (rad) tính từ trục dọc xe
          Trái  = góc dương (counter-clockwise)
          Phải  = góc âm   (clockwise)
        F1Tenth Hokuyo thường: angle_min = -2.35, angle_max = 2.35 rad
        """
        obs   = ObstacleInfo()
        ranges = np.array(msg.ranges, dtype=np.float32)
        n     = len(ranges)
        amin  = msg.angle_min
        ainc  = msg.angle_increment

        # Thay NaN/Inf bằng max_range để không gây lỗi
        max_r = msg.range_max if msg.range_max > 0 else 10.0
        ranges = np.where(np.isfinite(ranges) & (ranges > msg.range_min), ranges, max_r)

        def sector_min(a_lo: float, a_hi: float) -> float:
            """Lấy min distance trong sector [a_lo, a_hi] rad."""
            i_lo = max(0, int((a_lo - amin) / ainc))
            i_hi = min(n - 1, int((a_hi - amin) / ainc))
            if i_lo > i_hi:
                return max_r
            return float(np.min(ranges[i_lo:i_hi + 1]))

        obs.d_front       = sector_min(-0.175, +0.175)   # ±10°
        obs.d_front_left  = sector_min(+0.175, +1.047)   # +10° → +60°
        obs.d_front_right = sector_min(-1.047, -0.175)   # -60° → -10°
        obs.d_left        = sector_min(+1.047, +2.094)   # +60° → +120°
        obs.d_right       = sector_min(-2.094, -1.047)   # -120° → -60°
        return obs

    # ────────────────────────────────────────────────────────────────────────
    # LỚP 1 – Trọng số Q động (ý tưởng gốc của bạn, cải tiến)
    # ────────────────────────────────────────────────────────────────────────

    def _layer1_weight_scaling(self, obs: ObstacleInfo):
        """
        Tăng Q_y về phía có vật cản.
        Xe sẽ "sợ" lệch sang phía nguy hiểm hơn → tự nhiên dạt ra.

        Công thức:  w(d) = q_obs_max * max(0, 1 - d/d_safe)^2
        Quadratic falloff: can thiệp mạnh khi gần, mờ dần khi xa.
        """
        def w(d: float, d_safe: float) -> float:
            ratio = max(0.0, 1.0 - d / d_safe)
            return self.q_obs_max * ratio * ratio

        # Lấy vật cản nguy hiểm nhất mỗi bên (min của front + side)
        d_danger_left  = min(obs.d_front_left,  obs.d_left)
        d_danger_right = min(obs.d_front_right, obs.d_right)

        obs.q_weight_left  = w(d_danger_left,  self.d_safe_side)
        obs.q_weight_right = w(d_danger_right, self.d_safe_side)

    # ────────────────────────────────────────────────────────────────────────
    # LỚP 2 – Siết ràng buộc δ (hard constraint)
    # ────────────────────────────────────────────────────────────────────────

    def _layer2_constraint_tightening(self, obs: ObstacleInfo):
        """
        Thu hẹp biên góc lái về phía nguy hiểm.

        Ví dụ: vật cản phía trước-phải (d_front_right < d_danger)
          → delta_max_eff giảm dần về 0 (cấm quẹo phải)
          → xe chỉ còn được quẹo trái hoặc đi thẳng

        Tỉ lệ thu hẹp: linear từ 1.0 (an toàn) → shrink_min (sát vật)
        """
        def shrink_ratio(d: float, d_safe: float) -> float:
            if d >= d_safe:
                return 1.0
            ratio = d / d_safe
            # Giữ tối thiểu shrink_min để không khoá cứng hoàn toàn
            return max(self.constraint_shrink_min, ratio)

        # Phía phải nguy hiểm → thu hẹp delta_max (bẻ phải ít lại)
        r_right = shrink_ratio(obs.d_front_right, self.d_danger)
        obs.delta_max_eff = self.delta_max * r_right

        # Phía trái nguy hiểm → thu hẹp |delta_min| (bẻ trái ít lại)
        r_left = shrink_ratio(obs.d_front_left, self.d_danger)
        obs.delta_min_eff = self.delta_min * r_left   # delta_min âm nên * ratio → ít âm hơn

        # Emergency: vật cản thẳng trước
        obs.emergency_brake = obs.d_front < self.d_stop

    # ────────────────────────────────────────────────────────────────────────
    # LỚP 3 – Dịch reference trajectory (proactive)
    # ────────────────────────────────────────────────────────────────────────

    def _layer3_ref_offset(self, obs: ObstacleInfo):
        """
        Dịch ngang toàn bộ reference trajectory tránh xa vật cản.
        MPC sẽ "muốn" đi đến điểm đã dịch → chủ động tránh từ xa.

        offset > 0 = dịch sang trái (trục xe), < 0 = dịch sang phải.
        Offset tỉ lệ nghịch với khoảng cách đến vật cản nguy hiểm nhất.
        """
        d_L = min(obs.d_front_left,  obs.d_left)
        d_R = min(obs.d_front_right, obs.d_right)

        def offset_magnitude(d: float) -> float:
            if d >= self.d_safe_side:
                return 0.0
            return self.ref_offset_max * (1.0 - d / self.d_safe_side)

        offset_from_left  = -offset_magnitude(d_L)  # vật cản trái → dịch phải (âm)
        offset_from_right = +offset_magnitude(d_R)  # vật cản phải → dịch trái (dương)

        # Superposition, clip trong ±ref_offset_max
        raw = offset_from_left + offset_from_right
        obs.ref_lateral_offset = float(np.clip(raw, -self.ref_offset_max, self.ref_offset_max))

    def _compute_speed_limit(self, obs: ObstacleInfo):
        """Giới hạn tốc độ tuyến tính theo khoảng cách phía trước."""
        if obs.d_front < self.d_safe_front:
            ratio = max(0.0, obs.d_front / self.d_safe_front)
            obs.v_limit_obs = self.v_min + ratio * (self.v_max - self.v_min)
        else:
            obs.v_limit_obs = self.v_max

    # ════════════════════════════════════════════════════════════════════════
    # CALLBACK ODOMETRY – vòng lặp điều khiển chính
    # ════════════════════════════════════════════════════════════════════════

    def odom_callback(self, msg: Odometry):
        if not self.waypoints:
            return

        now = self.get_clock().now()
        if (now - self.last_mpc_time).nanoseconds / 1e9 < self.Ts:
            return
        self.last_mpc_time = now

        obs = self.obstacle_info   # snapshot LiDAR mới nhất

        # ── Emergency brake ──────────────────────────────────────────────
        if obs.emergency_brake:
            self._publish_drive(0.0, 0.0)
            self.get_logger().warn(
                f"[EMERGENCY] Vật cản {obs.d_front:.2f}m – dừng xe!")
            return

        v_x      = msg.twist.twist.linear.x
        v_y      = msg.twist.twist.linear.y
        yaw_rate = msg.twist.twist.angular.z
        self.v_x_current = max(v_x, self.v_min)

        # ── Pose từ TF ───────────────────────────────────────────────────
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.car_frame, rclpy.time.Time(seconds=0))
            rx    = tf.transform.translation.x
            ry    = tf.transform.translation.y
            r_yaw = euler_from_quaternion(
                tf.transform.rotation.x, tf.transform.rotation.y,
                tf.transform.rotation.z, tf.transform.rotation.w)
        except TransformException:
            return

        nearest_idx = self._find_nearest_waypoint(rx, ry)
        wp_x, wp_y, wp_yaw = self.waypoints[nearest_idx]

        e_psi = normalize_angle(r_yaw - wp_yaw)
        dx    = rx - wp_x
        dy    = ry - wp_y
        e_y   = -math.sin(wp_yaw) * dx + math.cos(wp_yaw) * dy

        # ── Tốc độ mục tiêu ──────────────────────────────────────────────
        v_curvature = self.compute_target_speed(nearest_idx)
        v_target    = min(v_curvature, obs.v_limit_obs)   # LiDAR giới hạn tốc độ

        # ── LỚP 1: Q động theo vật cản ───────────────────────────────────
        # Nếu vật cản bên phải → tăng phạt e_y dương (xe đang lệch phải)
        # Nếu vật cản bên trái  → tăng phạt e_y âm   (xe đang lệch trái)
        q_y_effective = (self.Q_base[1, 1]
                         + obs.q_weight_right  # phải nguy → phạt mạnh khi e_y<0
                         + obs.q_weight_left)  # trái  nguy → phạt mạnh khi e_y>0
        Q_eff = np.diag([self.Q_base[0, 0], q_y_effective])
        S_eff = np.diag([self.S_base[0, 0], q_y_effective])

        # ── State-space ───────────────────────────────────────────────────
        Ad, Bd, Cd = self.calculate_state_space(self.v_x_current)
        Hdb, Fdbt, Cdb, Adc = self.mpc_simplification(Ad, Bd, Cd, self.hz,
                                                        Q_eff, S_eff)

        states  = np.array([v_y, e_psi, yaw_rate, e_y, self.v_x_current])
        x_aug_t = np.concatenate((states, [self.U1, self.U2]))

        # ── LỚP 3: Reference với offset ngang ───────────────────────────
        r_vector, ref_pts = self._build_reference(
            nearest_idx, wp_x, wp_y, wp_yaw,
            lateral_offset=obs.ref_lateral_offset)

        self.publish_mpc_reference(ref_pts)

        # ── QP ─────────────────────────────────────────────────────────
        ft       = (Fdbt.T @ np.concatenate((x_aug_t, r_vector))).astype(np.float64)
        Hdb_f64  = 0.5 * (Hdb + Hdb.T) + 1e-8 * np.eye(Hdb.shape[0])

        du = self._solve(Hdb_f64, ft, obs)   # Lớp 2 passed vào _solve
        if du is None:
            return

        self.U1 = float(np.clip(
            self.U1 + du[0], obs.delta_min_eff, obs.delta_max_eff))   # Lớp 2
        self.U2 = float(np.clip(self.U2 + du[1], self.a_min, self.a_max))

        self.v_x_current = float(np.clip(
            self.v_x_current + self.U2 * self.Ts, self.v_min, self.v_max))

        self._publish_drive(self.U1, v_target)

        # Debug log mỗi 2 giây
        if (now.nanoseconds // int(2e9)) % 2 == 0:
            self.get_logger().debug(
                f"d_F={obs.d_front:.2f} dFL={obs.d_front_left:.2f} "
                f"dFR={obs.d_front_right:.2f} | "
                f"Q_y={q_y_effective:.0f} "
                f"δ∈[{obs.delta_min_eff:.2f},{obs.delta_max_eff:.2f}] "
                f"offset={obs.ref_lateral_offset:.3f}m")

    # ════════════════════════════════════════════════════════════════════════
    # STATE-SPACE (Bilinear, 5 states, 2 inputs)
    # ════════════════════════════════════════════════════════════════════════

    def calculate_state_space(self, v_x: float):
        v_x = max(v_x, 1.0)
        A1 = -(2*self.Caf + 2*self.Car)                            / (self.m  * v_x)
        A2 = -v_x - (2*self.Caf*self.lf - 2*self.Car*self.lr)     / (self.m  * v_x)
        A3 = -(2*self.lf*self.Caf - 2*self.lr*self.Car)           / (self.Iz * v_x)
        A4 = -(2*self.lf**2*self.Caf + 2*self.lr**2*self.Car)     / (self.Iz * v_x)

        A_c = np.array([
            [A1, 0, A2, 0, 0],
            [0,  0,  1, 0, 0],
            [A3, 0, A4, 0, 0],
            [1, v_x, 0, 0, 0],
            [0,  0,  0, 0, 0],
        ])
        B_c = np.array([
            [2*self.Caf/self.m,         0],
            [0,                         0],
            [2*self.lf*self.Caf/self.Iz, 0],
            [0,                         0],
            [0,                         1],
        ])
        C_c = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]])

        I        = np.eye(5)
        inv_term = np.linalg.inv(I - (self.Ts / 2.0) * A_c)
        Ad = inv_term @ (I + (self.Ts / 2.0) * A_c)
        Bd = inv_term @ (B_c * self.Ts)
        return Ad, Bd, C_c

    # ════════════════════════════════════════════════════════════════════════
    # MPC SIMPLIFICATION (nhận Q, S động từ Lớp 1)
    # ════════════════════════════════════════════════════════════════════════

    def mpc_simplification(self, Ad, Bd, Cd, hz, Q=None, S=None):
        """Xây Hdb, Fdbt với Q và S có thể thay đổi theo LiDAR (Lớp 1)."""
        Q = Q if Q is not None else self.Q_base
        S = S if S is not None else self.S_base

        n_x, n_u, n_y = Ad.shape[0], Bd.shape[1], Cd.shape[0]
        n_aug = n_x + n_u   # 7

        A_aug = np.block([[Ad, Bd], [np.zeros((n_u, n_x)), np.eye(n_u)]])
        B_aug = np.block([[Bd], [np.eye(n_u)]])
        C_aug = np.block([[Cd, np.zeros((n_y, n_u))]])

        CQC = C_aug.T @ Q @ C_aug
        CSC = C_aug.T @ S @ C_aug
        QC  = Q @ C_aug
        SC  = S @ C_aug

        s_x, s_y, s_u = n_aug*hz, n_y*hz, n_u*hz
        Qdb = np.zeros((s_x, s_x))
        Tdb = np.zeros((s_y, s_x))
        Rdb = np.zeros((s_u, s_u))
        Cdb = np.zeros((s_x, s_u))
        Adc = np.zeros((s_x, n_aug))

        for i in range(hz):
            if i == hz - 1:
                Qdb[n_aug*i:n_aug*(i+1), n_aug*i:n_aug*(i+1)] = CSC
                Tdb[n_y*i:n_y*(i+1),     n_aug*i:n_aug*(i+1)] = SC
            else:
                Qdb[n_aug*i:n_aug*(i+1), n_aug*i:n_aug*(i+1)] = CQC
                Tdb[n_y*i:n_y*(i+1),     n_aug*i:n_aug*(i+1)] = QC
            Rdb[n_u*i:n_u*(i+1), n_u*i:n_u*(i+1)] = self.R
            for j in range(i + 1):
                Cdb[n_aug*i:n_aug*(i+1), n_u*j:n_u*(j+1)] = (
                    np.linalg.matrix_power(A_aug, i - j) @ B_aug)
            Adc[n_aug*i:n_aug*(i+1), :] = np.linalg.matrix_power(A_aug, i + 1)

        Hdb  = Cdb.T @ Qdb @ Cdb + Rdb
        Fdbt = np.vstack((Adc.T @ Qdb @ Cdb, -Tdb @ Cdb))
        return Hdb, Fdbt, Cdb, Adc

    # ════════════════════════════════════════════════════════════════════════
    # XÂY MA TRẬN RÀNG BUỘC (Lớp 2 truyền vào delta_max/min_eff)
    # ════════════════════════════════════════════════════════════════════════

    def build_constraints(self, hz: int, obs: ObstacleInfo):
        """
        G * du <= ht   với   du = [dδ_0, da_0, dδ_1, da_1, ...]

        Lớp 2: dùng delta_max_eff / delta_min_eff từ ObstacleInfo
        thay vì delta_max / delta_min cố định.
        """
        I2 = np.eye(2)
        n2 = 2 * hz
        # Rate limit block
        Ib = np.zeros((n2, n2))
        for k in range(hz):
            Ib[2*k:2*k+2, 2*k:2*k+2] = I2
        rate_max = np.tile([self.du_delta_max, self.du_a_max], hz)

        # Cumulative sum block
        L = np.zeros((n2, n2))
        for i in range(hz):
            for j in range(i + 1):
                L[2*i:2*i+2, 2*j:2*j+2] = I2

        # ── Lớp 2: biên lái hiệu dụng từ LiDAR ─────────────────────────
        U_max_vec = np.tile([obs.delta_max_eff, self.a_max],  hz)
        U_min_vec = np.tile([obs.delta_min_eff, self.a_min],  hz)
        U_curr    = np.tile([self.U1, self.U2], hz)

        G  = np.vstack([ Ib, -Ib,  L, -L])
        ht = np.concatenate([rate_max, rate_max,
                             U_max_vec - U_curr,
                             U_curr    - U_min_vec])
        return G.astype(np.float64), ht.astype(np.float64)

    # ════════════════════════════════════════════════════════════════════════
    # QP SOLVER
    # ════════════════════════════════════════════════════════════════════════

    def _solve(self, Hdb, ft, obs: ObstacleInfo):
        if QP_AVAILABLE:
            G, ht = self.build_constraints(self.hz, obs)
            try:
                du = solve_qp(Hdb, ft, G=G, h=ht, solver="osqp",
                              verbose=False, eps_abs=1e-5, eps_rel=1e-5)
                if du is None:
                    raise ValueError("solve_qp → None")
                return du
            except Exception as e:
                self.get_logger().warn(
                    f"[QP fail] {e} | cond={np.linalg.cond(Hdb):.1e} "
                    f"d_front={obs.d_front:.2f}")
                return self._solve_unconstrained(Hdb, ft)
        return self._solve_unconstrained(Hdb, ft)

    def _solve_unconstrained(self, Hdb, ft):
        try:
            return -np.linalg.solve(Hdb, ft)
        except np.linalg.LinAlgError:
            return None

    # ════════════════════════════════════════════════════════════════════════
    # TỐC ĐỘ THÍCH NGHI THEO ĐỘ CONG
    # ════════════════════════════════════════════════════════════════════════

    def compute_target_speed(self, idx: int) -> float:
        n  = len(self.waypoints)
        k  = self.curvature_lookahead
        pp = self.waypoints[(idx - k) % n]
        pc = self.waypoints[idx]
        pn = self.waypoints[(idx + k) % n]
        dx1, dy1 = pc[0]-pp[0], pc[1]-pp[1]
        dx2, dy2 = pn[0]-pc[0], pn[1]-pc[1]
        cross     = abs(dx1*dy2 - dy1*dx2)
        curvature = cross / (math.hypot(dx1,dy1)*math.hypot(dx2,dy2) + 1e-6)
        spd = self.v_straight + min(curvature,1.0) * (self.v_curve - self.v_straight)
        return float(np.clip(spd, self.v_min, self.v_max))

    # ════════════════════════════════════════════════════════════════════════
    # REFERENCE TRAJECTORY (Lớp 3: thêm lateral_offset)
    # ════════════════════════════════════════════════════════════════════════

    def _build_reference(self, nearest_idx, wp_x, wp_y, wp_yaw,
                         lateral_offset: float = 0.0):
        """
        Tính reference vector.
        lateral_offset (m): dịch ngang toàn bộ điểm nhìn trước ra khỏi vật cản.
          > 0 = dịch trái (sang phía e_y dương trong toạ độ cục bộ)
          < 0 = dịch phải
        """
        r_list, ref_pts = [], []
        step_dist  = max(self.v_x_current, 1.0) * self.Ts
        curr_idx   = nearest_idx
        dist_accum = 0.0

        for i in range(1, self.hz + 1):
            target_dist = i * step_dist
            while True:
                nxt  = (curr_idx + 1) % len(self.waypoints)
                p1, p2 = self.waypoints[curr_idx], self.waypoints[nxt]
                seg   = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                if dist_accum + seg >= target_dist:
                    ratio = (target_dist - dist_accum) / seg if seg > 0 else 0.0
                    fx    = p1[0] + ratio*(p2[0]-p1[0])
                    fy    = p1[1] + ratio*(p2[1]-p1[1])
                    fyaw  = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
                    break
                dist_accum += seg
                curr_idx    = nxt

            ref_pts.append((fx, fy))
            fdx = fx - wp_x
            fdy = fy - wp_y
            # Lớp 3: cộng lateral_offset vào e_y_ref
            # local_y âm = xe đang ở bên phải track → tăng local_y → dịch trái
            local_y   = (-math.sin(wp_yaw)*fdx + math.cos(wp_yaw)*fdy) + lateral_offset
            local_yaw = normalize_angle(fyaw - wp_yaw)
            r_list.extend([local_yaw, local_y])

        return np.array(r_list), ref_pts

    # ════════════════════════════════════════════════════════════════════════
    # WAYPOINT LOADING & PATH SMOOTHING
    # ════════════════════════════════════════════════════════════════════════

    def load_waypoints(self, filename: str):
        raw = []
        try:
            with open(filename, 'r') as f:
                for row in csv.reader(f):
                    if not row: continue
                    cols = row[0].split() if len(row) == 1 else row
                    try:    raw.append([float(cols[0]), float(cols[1])])
                    except: continue
            smoothed = self.smooth_path(raw)
            self.waypoints = []
            for i in range(len(smoothed)):
                p1  = smoothed[i]
                p2  = smoothed[(i+1) % len(smoothed)]
                yaw = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
                self.waypoints.append([p1[0], p1[1], yaw])
        except Exception as e:
            self.get_logger().error(f"Lỗi đọc CSV: {e}")

    def smooth_path(self, path, wd=0.5, ws=0.2, tol=1e-5):
        p = deepcopy(path)
        change = tol
        while change >= tol:
            change = 0.0
            for i in range(1, len(path)-1):
                ax, ay = p[i][0], p[i][1]
                p[i][0] += wd*(path[i][0]-p[i][0]) + ws*(p[i-1][0]+p[i+1][0]-2*p[i][0])
                p[i][1] += wd*(path[i][1]-p[i][1]) + ws*(p[i-1][1]+p[i+1][1]-2*p[i][1])
                change  += abs(ax-p[i][0]) + abs(ay-p[i][1])
        return p

    # ════════════════════════════════════════════════════════════════════════
    # HELPER
    # ════════════════════════════════════════════════════════════════════════

    def _find_nearest_waypoint(self, rx: float, ry: float) -> int:
        if self.start_index is None:
            dists = [math.hypot(rx-p[0], ry-p[1]) for p in self.waypoints]
            self.start_index = int(np.argmin(dists))
            return self.start_index
        idx   = self.start_index
        cur_d = math.hypot(rx-self.waypoints[idx][0], ry-self.waypoints[idx][1])
        for _ in range(20):
            nxt   = (idx+1) % len(self.waypoints)
            nxt_d = math.hypot(rx-self.waypoints[nxt][0], ry-self.waypoints[nxt][1])
            if nxt_d < cur_d:
                idx, cur_d = nxt, nxt_d
            else:
                break
        self.start_index = idx
        return idx

    def _publish_drive(self, steering: float, speed: float):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed          = float(speed)
        self.pub_drive.publish(msg)

    # ── MARKERS ──────────────────────────────────────────────────────────

    def publish_mpc_reference(self, pts):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns, m.id        = "mpc_ref", 0
        m.type            = Marker.SPHERE_LIST
        m.action          = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.15
        m.color.a = 1.0; m.color.g = 1.0; m.color.b = 1.0
        for pt in pts:
            p = Point(); p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.1
            m.points.append(p)
        self.pub_mpc_ref.publish(m)

    def _publish_obstacle_marker(self, obs: ObstacleInfo):
        """Vẽ sphere đỏ ở phía trước xe nếu vật cản gần (RViz debug)."""
        m = Marker()
        m.header.frame_id = self.car_frame
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns, m.id        = "obs_zone", 0
        m.type            = Marker.SPHERE
        m.action          = Marker.ADD
        m.pose.position.x = obs.d_front
        m.pose.position.y = 0.0
        m.pose.position.z = 0.1
        m.pose.orientation.w = 1.0
        r = max(0.1, min(0.5, 1.0 - obs.d_front / self.d_safe_front))
        m.scale.x = m.scale.y = m.scale.z = r
        m.color.a = 0.6
        m.color.r = 1.0 if obs.d_front < self.d_danger else 0.8
        m.color.g = 0.0 if obs.d_front < self.d_danger else 0.5
        m.color.b = 0.0
        self.pub_obs_marker.publish(m)

    def publish_full_waypoint(self):
        arr = MarkerArray()
        m   = Marker()
        m.header.frame_id = "map"
        m.header.stamp    = self.get_clock().now().to_msg()
        m.id = 0; m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.a = 1.0; m.color.r = 1.0; m.color.g = 1.0
        for wp in self.waypoints:
            p = Point(); p.x, p.y, p.z = float(wp[0]), float(wp[1]), 0.0
            m.points.append(p)
        if self.waypoints:
            p = Point(); p.x, p.y = float(self.waypoints[0][0]), float(self.waypoints[0][1])
            m.points.append(p)
        arr.markers.append(m)
        self.pub_marker_path.publish(arr)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MPCNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback; traceback.print_exc()
    finally:
        if node:
            stop = AckermannDriveStamped()
            stop.drive.speed = 0.0
            node.pub_drive.publish(stop)
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
