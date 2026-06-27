#!/usr/bin/env python3
"""
MPC Controller cho Xe Mô Phỏng – F1Tenth Simulator Deployment
============================================================
Kiến trúc: Timer-driven | Kinematic Bicycle Model (LPV) | Delay Compensation
Cấu hình: Đã đồng bộ hóa sang các thông số hình học và Topic của F1Tenth Simulator.
"""

import rclpy
from rclpy.node import Node
import scipy.sparse as sparse
import math
import csv
import time
import numpy as np
from copy import deepcopy
import threading

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


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================
def euler_from_quaternion(x, y, z, w):
    """Trích yaw từ quaternion (không cần scipy/tf_transformations)."""
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def normalize_angle(angle):
    """Đưa góc về [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ======================================================================
# MPC NODE
# ======================================================================
class MPCRealCarNode(Node):

    def __init__(self):
        super().__init__('mpc_simulator_node')

        # ==============================================================
        # 1. KHAI BÁO THAM SỐ ROS2 - ĐÃ HIỆU CHỈNH CHO MÔ PHỎNG
        # ==============================================================
        # ── Vật lý xe mô phỏng ──
        self.declare_parameter('lf', 0.158)         # Tâm KL → trục trước (m)
        self.declare_parameter('lr', 0.171)         # Tâm KL → trục sau  (m)

        # ── MPC core ──
        self.declare_parameter('Ts', 0.05)          # Chu kỳ lấy mẫu (s) - 20Hz
        self.declare_parameter('hz', 20)             # Prediction horizon (steps)

        # ── Trọng số Q (running), S (terminal), R (input rate) ──
        self.declare_parameter('Q_epsi',  20.0)     # Phạt lỗi góc hướng
        self.declare_parameter('Q_ey',   300.0)     # Phạt lỗi ngang
        self.declare_parameter('Q_vx',   150.0)     # Phạt lệch tốc độ
        self.declare_parameter('S_epsi',  20.0)
        self.declare_parameter('S_ey',   300.0)
        self.declare_parameter('S_vx',   150.0)
        self.declare_parameter('R_delta', 5000.0)   # Phạt thay đổi góc lái
        self.declare_parameter('R_accel', 100.0)    # Phạt thay đổi gia tốc

        # ── Ràng buộc vật lý mô phỏng ──
        self.declare_parameter('delta_max',    0.35)   # rad 
        self.declare_parameter('du_delta_max', 0.08)   # rad/step
        self.declare_parameter('a_max',        3.0)    # m/s²
        self.declare_parameter('a_min',       -4.0)    # m/s²
        self.declare_parameter('du_a_max',     1.0)    # m/s³
        self.declare_parameter('v_max',        9.5)    # m/s (Có thể nâng cao trong sim)
        self.declare_parameter('v_min',        0.5)    # m/s

        # ── Tốc độ thích nghi theo độ cong ──
        self.declare_parameter('v_straight',          9.5)
        self.declare_parameter('v_curve',             1.5)
        self.declare_parameter('curvature_lookahead', 10)

        # ── Bù trễ & lọc (Hiệu chỉnh về 0 cho mô phỏng lý tưởng) ──
        self.declare_parameter('delay_steps',      0)      # Sim không có trễ phần cứng
        self.declare_parameter('v_x_cache_thresh', 0.15)
        self.declare_parameter('lp_alpha',         0.0)    # Không cần lọc thông thấp trong sim
        self.declare_parameter('watchdog_timeout', 0.50)

        # ── Topics & Frames (Đồng bộ chuẩn thông số của bạn) ──
        self.declare_parameter('csv_path',
            '/sim_ws/install/waypoint/share/waypoint/'
            'f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv')
        self.declare_parameter('car_frame',  'ego_racecar/base_link')
        self.declare_parameter('map_frame',  'map')
        self.declare_parameter('odom_topic', 'ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')

        # ==============================================================
        # 2. ĐỌC THAM SỐ VÀO HỆ THỐNG
        # ==============================================================
        self.lf  = self.get_parameter('lf').value
        self.lr  = self.get_parameter('lr').value
        self.L   = self.lf + self.lr
        self.Ts  = self.get_parameter('Ts').value
        self.hz  = self.get_parameter('hz').value

        self.Q = np.diag([
            self.get_parameter('Q_epsi').value,
            self.get_parameter('Q_ey').value,
            self.get_parameter('Q_vx').value,
        ])
        self.S = np.diag([
            self.get_parameter('S_epsi').value,
            self.get_parameter('S_ey').value,
            self.get_parameter('S_vx').value,
        ])
        self.R = np.diag([
            self.get_parameter('R_delta').value,
            self.get_parameter('R_accel').value,
        ])

        self.delta_max        = self.get_parameter('delta_max').value
        self.delta_min        = -self.delta_max
        self.du_delta_max     = self.get_parameter('du_delta_max').value
        self.a_max            = self.get_parameter('a_max').value
        self.a_min            = self.get_parameter('a_min').value
        self.du_a_max         = self.get_parameter('du_a_max').value
        self.v_max            = self.get_parameter('v_max').value
        self.v_min            = self.get_parameter('v_min').value
        self.v_straight       = self.get_parameter('v_straight').value
        self.v_curve          = self.get_parameter('v_curve').value
        self.curvature_lookahead = self.get_parameter('curvature_lookahead').value

        self.delay_steps      = max(0, self.get_parameter('delay_steps').value)
        self.v_x_cache_thresh = self.get_parameter('v_x_cache_thresh').value
        self.lp_alpha         = self.get_parameter('lp_alpha').value
        self.watchdog_timeout = self.get_parameter('watchdog_timeout').value

        self.car_frame = self.get_parameter('car_frame').value
        self.map_frame = self.get_parameter('map_frame').value

        # ==============================================================
        # 3. KHỞI TẠO BIẾN TRẠNG THÁI
        # ==============================================================
        self.state_lock     = threading.Lock()
        self.v_x_filtered   = 0.0
        self.current_v_x    = 0.0
        self.last_odom_time = None

        # Khởi tạo buffer dựa trên số bước trễ cấu hình
        self.u_buffer = [(0.0, 0.0)] * max(1, self.delay_steps)

        self.U1 = 0.0   # delta tích lũy
        self.U2 = 0.0   # acceleration tích lũy

        self._cached_v_x   = None
        self._cached_Hdb   = None
        self._cached_Fdbt  = None
        self._cached_Cdb   = None
        self._cached_Adc   = None

        self._G_sparse  = None
        self._rate_max  = None
        self._U_max_vec = None
        self._U_min_vec = None
        self._build_constraint_matrices()

        self.waypoints   = []
        self.start_index = None

        # ==============================================================
        # 4. ROS 2 INTERFACES
        # ==============================================================
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        odom_topic  = self.get_parameter('odom_topic').value
        drive_topic = self.get_parameter('drive_topic').value

        self.sub_odom    = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 5)
        self.pub_drive   = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10)
        self.pub_wp_path = self.create_publisher(
            MarkerArray, '/publish_full_waypoint', 10)
        self.pub_mpc_ref = self.create_publisher(
            Marker, '/mpc_lookahead_points', 10)
        self.pub_predict = self.create_publisher(
            Marker, '/mpc_predict_path', 10)

        self.load_waypoints(self.get_parameter('csv_path').value)
        self.publish_full_waypoint()

        self.control_timer = self.create_timer(self.Ts, self.control_loop)

        if not QP_AVAILABLE:
            self.get_logger().error('[CRITICAL] Thiếu qpsolvers[osqp]!')
        
        self.get_logger().info(
            f'[MPC SIMULATOR] Sẵn sàng. Target frames: {self.map_frame} -> {self.car_frame}'
        )

    def odom_callback(self, msg: Odometry):
        raw_vx = msg.twist.twist.linear.x
        with self.state_lock:
            self.v_x_filtered = (self.lp_alpha * self.v_x_filtered
                                 + (1.0 - self.lp_alpha) * raw_vx)
            self.current_v_x  = self.v_x_filtered
            self.last_odom_time = self.get_clock().now()

    def calculate_kinematic_state_space(self, v_x):
        v_x = max(v_x, self.v_min)
        Ad = np.array([
            [1.0,           0.0, 0.0],
            [v_x * self.Ts, 1.0, 0.0],
            [0.0,           0.0, 1.0],
        ])
        Bd = np.array([
            [v_x * self.Ts / self.L,               0.0    ],
            [v_x**2 * self.Ts**2 / (2.0 * self.L), 0.0    ],
            [0.0,                                   self.Ts],
        ])
        Cd = np.eye(3)
        return Ad, Bd, Cd

    def _get_mpc_matrices(self, v_x):
        if (self._cached_v_x is not None
                and abs(v_x - self._cached_v_x) < self.v_x_cache_thresh):
            return (self._cached_Hdb, self._cached_Fdbt,
                    self._cached_Cdb,  self._cached_Adc)

        Ad, Bd, Cd = self.calculate_kinematic_state_space(v_x)
        result = self._mpc_simplification(Ad, Bd, Cd, self.hz)

        self._cached_v_x, self._cached_Hdb   = v_x,      result[0]
        self._cached_Fdbt, self._cached_Cdb  = result[1], result[2]
        self._cached_Adc                     = result[3]
        return result

    def _mpc_simplification(self, Ad, Bd, Cd, hz):
        n_x   = Ad.shape[0]
        n_u   = Bd.shape[1]
        n_y   = Cd.shape[0]
        n_aug = n_x + n_u

        A_aug = np.block([
            [Ad,                    Bd         ],
            [np.zeros((n_u, n_x)), np.eye(n_u) ],
        ])
        B_aug = np.block([[Bd], [np.eye(n_u)]])
        C_aug = np.block([[Cd, np.zeros((n_y, n_u))]])

        CQC = C_aug.T @ self.Q @ C_aug
        CSC = C_aug.T @ self.S @ C_aug
        QC  = self.Q @ C_aug
        SC  = self.S @ C_aug

        A_pwr = [np.eye(n_aug)]
        for _ in range(hz):
            A_pwr.append(A_pwr[-1] @ A_aug)

        s_x = n_aug * hz
        s_y = n_y   * hz
        s_u = n_u   * hz

        Qdb = np.zeros((s_x, s_x))
        Tdb = np.zeros((s_y, s_x))
        Rdb = np.kron(np.eye(hz), self.R)
        Cdb = np.zeros((s_x, s_u))
        Adc = np.zeros((s_x, n_aug))

        for i in range(hz):
            Q_blk = CSC if i == hz - 1 else CQC
            T_blk = SC  if i == hz - 1 else QC

            Qdb[n_aug*i : n_aug*(i+1), n_aug*i : n_aug*(i+1)] = Q_blk
            Tdb[n_y *i  : n_y *(i+1), n_aug*i : n_aug*(i+1)] = T_blk
            Adc[n_aug*i : n_aug*(i+1), :]                     = A_pwr[i + 1]

            for j in range(i + 1):
                Cdb[n_aug*i : n_aug*(i+1), n_u*j : n_u*(j+1)] = (
                    A_pwr[i - j] @ B_aug
                )

        Hdb  = Cdb.T @ Qdb @ Cdb + Rdb
        Hdb  = (0.5 * (Hdb + Hdb.T) + 1e-8 * np.eye(s_u)).astype(np.float64)
        Fdbt = np.vstack([Adc.T @ Qdb @ Cdb, -Tdb @ Cdb])

        return Hdb, Fdbt, Cdb, Adc

    def _build_constraint_matrices(self):
        hz     = self.hz
        n2     = 2 * hz
        L_kron = np.kron(np.tril(np.ones((hz, hz))), np.eye(2))
        G      = np.vstack([np.eye(n2), -np.eye(n2), L_kron, -L_kron]).astype(np.float64)

        self._G_sparse  = sparse.csc_matrix(G)
        self._rate_max  = np.tile([self.du_delta_max, self.du_a_max], hz)
        self._U_max_vec = np.tile([self.delta_max, self.a_max],  hz)
        self._U_min_vec = np.tile([self.delta_min, self.a_min],  hz)

    def _build_ht(self):
        U_curr = np.tile([self.U1, self.U2], self.hz)
        return np.concatenate([
            self._rate_max,
            self._rate_max,
            self._U_max_vec - U_curr,
            U_curr - self._U_min_vec,
        ])

    def _solve(self, Hdb, ft):
        if not QP_AVAILABLE:
            try: return -np.linalg.solve(Hdb, ft)
            except np.linalg.LinAlgError: return None

        ht = self._build_ht()
        try:
            du = solve_qp(
                sparse.csc_matrix(Hdb), ft.astype(np.float64),
                G=self._G_sparse, h=ht.astype(np.float64),
                solver='osqp', verbose=False,
                eps_abs=1e-5, eps_rel=1e-5, max_iter=4000,
            )
            return du
        except Exception:
            try: return -np.linalg.solve(Hdb, ft)
            except np.linalg.LinAlgError: return None

    def control_loop(self):
        if not self.waypoints:
            return

        need_stop = False
        with self.state_lock:
            if self.last_odom_time is None:
                return
            dt_odom = (self.get_clock().now() - self.last_odom_time).nanoseconds / 1e9
            if dt_odom > self.watchdog_timeout:
                need_stop = True
            else:
                v_x_curr = max(self.current_v_x, self.v_min)

        if need_stop:
            self._emergency_stop()
            return

        # ── 1. Lấy pose từ TF ────────────────────────────────────────
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.car_frame, rclpy.time.Time())
            rx    = tf.transform.translation.x
            ry    = tf.transform.translation.y
            q     = tf.transform.rotation
            r_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        except TransformException as e:
            return

        # ── 2. Waypoint gần nhất & tracking error ────────────────────
        nearest_idx        = self._find_nearest_waypoint(rx, ry)
        wp_x, wp_y, wp_yaw = self.waypoints[nearest_idx]

        e_psi_raw = normalize_angle(r_yaw - wp_yaw)
        dx        = rx - wp_x
        dy        = ry - wp_y
        e_y_raw   = -math.sin(wp_yaw) * dx + math.cos(wp_yaw) * dy

        # ── 3. BÙ TRỄ HỆ THỐNG (Chỉ dự đoán nếu delay_steps > 0) ──────
        x_state = np.array([e_psi_raw, e_y_raw, v_x_curr])
        if self.delay_steps > 0:
            for step in range(self.delay_steps):
                ud = self.u_buffer[step]
                Ad_d, Bd_d, _ = self.calculate_kinematic_state_space(x_state[2])
                x_state = Ad_d @ x_state + Bd_d @ np.array(ud)

        x_pred   = x_state
        v_x_pred = max(x_pred[2], self.v_min)

        # ── 4. Xây MPC matrices ───────────────────────────────────────
        Hdb, Fdbt, Cdb, Adc = self._get_mpc_matrices(v_x_pred)
        x_aug_t = np.concatenate((x_pred, [self.U1, self.U2]))

        # ── 5. Reference trajectory ───────────────────────────────────
        r_vector, ref_pts, v_target_now = self._build_reference(
            nearest_idx, wp_x, wp_y, wp_yaw, v_x_pred)
        self.publish_mpc_reference(ref_pts)

        # ── 6. Giải QP ────────────────────────────────────────────────
        ft_input = np.concatenate((x_aug_t, r_vector)).astype(np.float64)
        ft       = Fdbt.T @ ft_input
        du = self._solve(Hdb, ft)

        if du is None:
            self._safe_brake(v_x_curr)
            return

        # ── 7. Cập nhật điều khiển tích lũy ──────────────────────────
        self.U1 = float(np.clip(self.U1 + du[0], self.delta_min, self.delta_max))
        self.U2 = float(np.clip(self.U2 + du[1], self.a_min,     self.a_max    ))

        # ── 8. Steering ──────────────────────────────────────────────
        steering_cmd = self.U1

        # ── 9. Delay buffer FIFO ──────────────────────────────────────
        if self.delay_steps > 0:
            self.u_buffer.pop(0)
            self.u_buffer.append((self.U1, self.U2))

        # ── 10. Tốc độ lệnh ──────────────────────────────────────────
        v_cmd = float(np.clip(v_x_curr + self.U2 * self.Ts, self.v_min, self.v_max))

        # ── 11. Publish drive ─────────────────────────────────────────
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp         = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = steering_cmd
        drive_msg.drive.speed          = v_cmd
        self.pub_drive.publish(drive_msg)

        # ── 12. Predicted trajectory ──────────────────────────────────
        X_pred_vec = Adc @ x_aug_t + Cdb @ du
        self._publish_predicted_path(X_pred_vec, wp_x, wp_y, wp_yaw)

    def compute_target_speed(self, idx):
        n      = len(self.waypoints)
        k      = self.curvature_lookahead
        p_prev = self.waypoints[(idx - k) % n]
        p_curr = self.waypoints[idx]
        p_next = self.waypoints[(idx + k) % n]

        dx1 = p_curr[0] - p_prev[0]; dy1 = p_curr[1] - p_prev[1]
        dx2 = p_next[0] - p_curr[0]; dy2 = p_next[1] - p_curr[1]
        cross     = abs(dx1 * dy2 - dy1 * dx2)
        n1        = math.hypot(dx1, dy1) + 1e-6
        n2        = math.hypot(dx2, dy2) + 1e-6
        sin_angle = min(cross / (n1 * n2), 1.0)

        return float(np.clip(
            self.v_straight + sin_angle * (self.v_curve - self.v_straight),
            self.v_min, self.v_max
        ))

    def _build_reference(self, nearest_idx, wp_x, wp_y, wp_yaw, current_v_x):
        r_list, ref_pts = [], []
        step_dist  = max(current_v_x, self.v_straight) * self.Ts
        curr_idx   = nearest_idx
        dist_accum = 0.0
        max_iter   = len(self.waypoints) + 1
        v_target_0 = None

        for i in range(1, self.hz + 1):
            target_dist = i * step_dist
            safe_count  = 0
            found       = False

            while safe_count < max_iter:
                nxt  = (curr_idx + 1) % len(self.waypoints)
                p1   = self.waypoints[curr_idx]
                p2   = self.waypoints[nxt]
                slen = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

                if slen < 1e-6:
                    curr_idx = nxt; safe_count += 1; continue

                if dist_accum + slen >= target_dist:
                    r    = np.clip((target_dist - dist_accum) / slen, 0.0, 1.0)
                    fx   = p1[0] + r * (p2[0] - p1[0])
                    fy   = p1[1] + r * (p2[1] - p1[1])
                    fyaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    found = True; break
                else:
                    dist_accum += slen
                    curr_idx    = nxt
                    safe_count += 1

            if not found:
                p1 = self.waypoints[curr_idx]
                fx, fy, fyaw = p1[0], p1[1], p1[2]

            ref_pts.append((fx, fy))
            ly   = -math.sin(wp_yaw) * (fx - wp_x) + math.cos(wp_yaw) * (fy - wp_y)
            lyaw = normalize_angle(fyaw - wp_yaw)

            fvt = self.compute_target_speed(curr_idx)
            if v_target_0 is None:
                v_target_0 = fvt

            r_list.extend([lyaw, ly, fvt])

        v0 = v_target_0 if v_target_0 is not None else self.v_min
        return np.array(r_list), ref_pts, v0

    def _find_nearest_waypoint(self, rx, ry):
        if self.start_index is None:
            dists = [math.hypot(rx - p[0], ry - p[1]) for p in self.waypoints]
            self.start_index = int(np.argmin(dists))
            return self.start_index

        idx    = self.start_index
        curr_d = math.hypot(rx - self.waypoints[idx][0], ry - self.waypoints[idx][1])

        if curr_d > 2.0:
            dists = [math.hypot(rx - p[0], ry - p[1]) for p in self.waypoints]
            self.start_index = int(np.argmin(dists))
            return self.start_index

        for _ in range(30):
            nxt = (idx + 1) % len(self.waypoints)
            d   = math.hypot(rx - self.waypoints[nxt][0], ry - self.waypoints[nxt][1])
            if d < curr_d:
                idx = nxt; curr_d = d
            else:
                break

        self.start_index = idx
        return idx

    def _emergency_stop(self):
        self.U1 = 0.0; self.U2 = 0.0
        msg = AckermannDriveStamped()
        msg.header.stamp         = self.get_clock().now().to_msg()
        msg.drive.speed          = 0.0
        msg.drive.steering_angle = 0.0
        self.pub_drive.publish(msg)

    def _safe_brake(self, v_x_curr):
        self.U1 *= 0.85
        self.U2  = max(self.U2 - 0.5, self.a_min)
        v_safe   = float(np.clip(v_x_curr + self.U2 * self.Ts, 0.0, self.v_max))

        msg = AckermannDriveStamped()
        msg.header.stamp         = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(self.U1)
        msg.drive.speed          = v_safe
        self.pub_drive.publish(msg)

    def load_waypoints(self, filename):
        raw = []
        try:
            with open(filename, 'r') as f:
                for row in csv.reader(f):
                    if not row or row[0].strip().startswith('#'):
                        continue
                    if len(row) == 1:
                        row = row[0].split()
                    try: raw.append([float(row[0]), float(row[1])])
                    except (ValueError, IndexError): continue

            if len(raw) < 4:
                return

            sm = self.smooth_path(raw)
            n  = len(sm)
            self.waypoints = [
                [sm[i][0], sm[i][1],
                 math.atan2(sm[(i+1) % n][1] - sm[i][1],
                            sm[(i+1) % n][0] - sm[i][0])]
                for i in range(n)
            ]
        except Exception as e:
            self.get_logger().error(f'[CSV Error] {e}')

    def smooth_path(self, path, weight_data=0.5, weight_smooth=0.2, tolerance=1e-5):
        new_path = deepcopy(path)
        change   = tolerance + 1.0
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path) - 1):
                for k in range(2):
                    old = new_path[i][k]
                    new_path[i][k] += (
                        weight_data   * (path[i][k] - new_path[i][k])
                        + weight_smooth * (new_path[i-1][k] + new_path[i+1][k]
                                           - 2.0 * new_path[i][k])
                    )
                    change += abs(old - new_path[i][k])
        return new_path

    def _publish_predicted_path(self, X_pred_vec, wp_x, wp_y, wp_yaw):
        n_aug  = 5
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.ns, marker.id   = 'mpc_predict', 1
        marker.type            = Marker.LINE_STRIP
        marker.action          = Marker.ADD
        marker.scale.x         = 0.08
        marker.color.a = 0.85; marker.color.r = 1.0

        s_accum = 0.0
        for i in range(self.hz):
            e_y_p = X_pred_vec[n_aug * i + 1]
            v_x_p = X_pred_vec[n_aug * i + 2]
            s_accum += max(v_x_p, self.v_min) * self.Ts
            px = wp_x + s_accum * math.cos(wp_yaw) - e_y_p * math.sin(wp_yaw)
            py = wp_y + s_accum * math.sin(wp_yaw) + e_y_p * math.cos(wp_yaw)
            marker.points.append(Point(x=float(px), y=float(py), z=0.15))
        self.pub_predict.publish(marker)

    def publish_mpc_reference(self, pts):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns, m.id        = 'mpc_ref', 0
        m.type            = Marker.SPHERE_LIST
        m.action          = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.15
        m.color.a = 1.0; m.color.g = 1.0; m.color.b = 1.0
        m.points  = [Point(x=float(p[0]), y=float(p[1]), z=0.1) for p in pts]
        self.pub_mpc_ref.publish(m)

    def publish_full_waypoint(self):
        mk = Marker()
        mk.header.frame_id = self.map_frame
        mk.header.stamp    = self.get_clock().now().to_msg()
        mk.id              = 0
        mk.type            = Marker.LINE_STRIP
        mk.action          = Marker.ADD
        mk.scale.x         = 0.05
        mk.color.a = 1.0; mk.color.r = 1.0; mk.color.g = 1.0
        mk.points = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in self.waypoints]
        if self.waypoints:
            mk.points.append(Point(x=float(self.waypoints[0][0]), y=float(self.waypoints[0][1]), z=0.0))
        arr = MarkerArray()
        arr.markers.append(mk)
        self.pub_wp_path.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MPCRealCarNode()
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if node is not None:
            node._emergency_stop()
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()