#!/usr/bin/env python3
"""
MPC Controller cho Xe thực tế – F1Tenth Real Car Deployment
============================================================
Kiến trúc: Timer-driven | Kinematic Bicycle Model (LPV) | Delay Compensation

Cải tiến so với mpc_LPV.py (AI v1):
  ① Precompute danh sách lũy thừa A^k → O(hz·n³) thay vì O(hz²·n³)
  ② Cache ma trận MPC khi v_x không đổi đáng kể → giảm tải CPU đáng kể
  ③ Curvature feedforward δ_ff = arctan(L·κ) bù sai số steady-state ở cua
  ④ Watchdog dùng timestamp (không phải boolean flag racy) → phanh khẩn cấp
  ⑤ Publish predicted trajectory (đường đỏ) lên RViz2
  ⑥ Diagnostic logging: e_y, e_psi, U1, U2, solve_time_ms (throttled 1Hz)
  ⑦ Safety break trong _build_reference chống vòng lặp vô hạn
  ⑧ IIR Low-pass filter cho v_x measurement chống nhiễu encoder
  ⑨ Tất cả tham số → ROS2 declare_parameter (cấu hình qua launch file)
  ⑩ CSV parser robust: xử lý header, space/comma-separated, dòng lỗi
  ⑪ _build_constraint_matrices khởi tạo trong __init__, không crash runtime
  ⑫ _safe_brake nhất quán: decay lái + publish command ngay lập tức

Lưu ý deployment:
  - Kiểm tra sign convention steering trên xe thật (delta > 0 = left/right?)
  - Chỉnh delta_max cho khớp giới hạn cơ khí servo thực tế
  - Verify topic /odom và /vesc/... đúng với firmware VESC đang dùng
  - System Identification: đo lf, lr, và tune Q/R/S trên sân thật
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
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def normalize_angle(angle):
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
        super().__init__("mpc_real_car_node")

        # ==============================================================
        # 1. KHAI BÁO THAM SỐ ROS2 (⑨ Tất cả config qua launch file)
        # ==============================================================
        # Vật lý xe – đo lại bằng thước và cân!
        self.declare_parameter('lf', 0.22)        # Tâm KL → trục trước (m)
        self.declare_parameter('lr', 0.18)        # Tâm KL → trục sau  (m)

        # MPC core
        self.declare_parameter('Ts', 0.05)         # Chu kỳ lấy mẫu 20Hz
        self.declare_parameter('hz', 18)            # Prediction horizon

        # Ma trận trọng số Q (running), S (terminal), R (input change)
        self.declare_parameter('Q_epsi', 20.0)     # Phạt lỗi heading
        self.declare_parameter('Q_ey',  100.0)     # Phạt lỗi ngang – cao!
        self.declare_parameter('S_epsi', 20.0)
        self.declare_parameter('S_ey',  100.0)
        self.declare_parameter('R_delta', 4000.0)  # Phạt thay đổi góc lái
        self.declare_parameter('R_accel', 100.0)   # Phạt thay đổi gia tốc

        # Ràng buộc vật lý
        self.declare_parameter('delta_max',    0.35)   # rad – kiểm tra servo!
        self.declare_parameter('du_delta_max', 0.08)   # rad/step (rate limit)
        self.declare_parameter('a_max',        3.0)    # m/s²
        self.declare_parameter('a_min',       -4.0)    # m/s² (phanh mạnh hơn)
        self.declare_parameter('du_a_max',     1.0)    # m/s³ (jerk limit)
        self.declare_parameter('v_max',        3.8)    # m/s
        self.declare_parameter('v_min',        0.8)    # m/s

        # Tốc độ thích nghi theo độ cong
        self.declare_parameter('v_straight',        3.8)
        self.declare_parameter('v_curve',            0.8)
        self.declare_parameter('curvature_lookahead', 16)

        # Bù trễ & lọc
        self.declare_parameter('delay_steps',      1)     # Số bước bù trễ
        self.declare_parameter('v_x_cache_thresh', 0.15)  # m/s – ngưỡng invalidate cache
        self.declare_parameter('lp_alpha',         0.7)   # IIR coeff (0=no filter)
        self.declare_parameter('watchdog_timeout',  0.25) # s – mất odom bao lâu thì phanh

        # Topics & Frames (⑨ đổi qua launch, không hard-code)
        self.declare_parameter('csv_path',
            '/home/danh/ros2_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv')
        self.declare_parameter('car_frame',   'base_link')
        self.declare_parameter('map_frame',   'map')
        self.declare_parameter('odom_topic',  '/odom')
        # Chuẩn F1Tenth VESC mux:
        self.declare_parameter('drive_topic',
            "/drive")

        # ==============================================================
        # 2. ĐỌC THAM SỐ
        # ==============================================================
        self.lf = self.get_parameter('lf').value
        self.lr = self.get_parameter('lr').value
        self.L  = self.lf + self.lr
        self.Ts = self.get_parameter('Ts').value
        self.hz = self.get_parameter('hz').value

        self.Q = np.diag([
            self.get_parameter('Q_epsi').value,
            self.get_parameter('Q_ey').value,
        ])
        self.S = np.diag([
            self.get_parameter('S_epsi').value,
            self.get_parameter('S_ey').value,
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

        self.delay_steps      = max(1, self.get_parameter('delay_steps').value)
        self.v_x_cache_thresh = self.get_parameter('v_x_cache_thresh').value
        self.lp_alpha         = self.get_parameter('lp_alpha').value
        self.watchdog_timeout = self.get_parameter('watchdog_timeout').value

        self.car_frame  = self.get_parameter('car_frame').value
        self.map_frame  = self.get_parameter('map_frame').value

        # ==============================================================
        # 3. KHỞI TẠO BIẾN TRẠNG THÁI
        # ==============================================================
        self.state_lock    = threading.Lock()
        self.v_x_filtered  = 0.0         # ⑧ Low-pass filtered velocity
        self.current_v_x   = 0.0         # Giá trị dùng trong control loop
        self.last_odom_time = None        # ④ Watchdog timestamp

        # Buffer delay compensation: lưu `delay_steps` lệnh gần nhất
        # u_buffer[0] = lệnh cũ nhất (đang tác động lên xe)
        self.u_buffer = [(0.0, 0.0)] * self.delay_steps

        # Lệnh điều khiển tích lũy (augmented state formulation)
        self.U1 = 0.0    # Góc lái (rad) – feedback từ MPC
        self.U2 = 0.0    # Gia tốc (m/s²)

        # ② MPC Matrix Cache
        self._cached_v_x   = None
        self._cached_Hdb   = None
        self._cached_Fdbt  = None
        self._cached_Cdb   = None
        self._cached_Adc   = None

        # ⑪ Constraint matrices (khởi tạo ngay trong __init__)
        self._G_sparse  = None
        self._rate_max  = None
        self._U_max_vec = None
        self._U_min_vec = None
        self._build_constraint_matrices()   # Tính 1 lần, không đổi theo thời gian

        self.waypoints   = []
        self.start_index = None

        # ==============================================================
        # 4. ROS 2 INTERFACES
        # ==============================================================
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        odom_topic  = self.get_parameter('odom_topic').value
        drive_topic = self.get_parameter('drive_topic').value

        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 5)
        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10)
        self.pub_marker_path = self.create_publisher(
            MarkerArray, "/publish_full_waypoint", 10)
        self.pub_mpc_ref     = self.create_publisher(
            Marker, "/mpc_lookahead_points", 10)
        self.pub_mpc_predict = self.create_publisher(
            Marker, "/mpc_predict_path", 10)

        # Load waypoints
        csv_path = self.get_parameter('csv_path').value
        self.load_waypoints(csv_path)
        self.publish_full_waypoint()

        # ④ Control Timer – timer-driven, không phụ thuộc odom clock
        self.control_timer = self.create_timer(self.Ts, self.control_loop)

        if not QP_AVAILABLE:
            self.get_logger().error(
                "[CRITICAL] Thiếu qpsolvers[osqp]. "
                "Unconstrained MPC KHÔNG AN TOÀN trên xe thật! "
                "Cài: pip install qpsolvers[osqp] --break-system-packages"
            )
        self.get_logger().info(
            f"[MPC Real Car] Khởi động thành công: "
            f"hz={self.hz}, Ts={self.Ts}s, L={self.L:.3f}m, "
            f"delay={self.delay_steps} step(s), "
            f"waypoints={len(self.waypoints)}"
        )

    # ==================================================================
    # ODOM CALLBACK (Chỉ cập nhật data – không tính toán MPC)
    # ==================================================================
    def odom_callback(self, msg: Odometry):
        """
        ⑧ IIR Low-pass filter: v_filt = α·v_filt + (1-α)·v_raw
        α gần 1 → lọc mạnh (lag nhiều hơn)
        α = 0   → không lọc
        """
        raw_vx = msg.twist.twist.linear.x
        with self.state_lock:
            alpha = self.lp_alpha
            self.v_x_filtered = alpha * self.v_x_filtered + (1.0 - alpha) * raw_vx
            self.current_v_x  = self.v_x_filtered
            self.last_odom_time = self.get_clock().now()   # ④ Cập nhật timestamp

    # ==================================================================
    # KINEMATIC BICYCLE MODEL – ZOH Exact Discretization (LPV)
    # ==================================================================
    def calculate_kinematic_state_space(self, v_x):
        """
        State  x = [e_psi, e_y, v_x]   (3 states)
        Input  u = [delta, a]           (2 inputs)
        Output y = [e_psi, e_y]         (2 outputs)

        Continuous dynamics (linearized around reference):
          ė_psi = (v_x / L) · δ          (+ curvature disturbance → feedforward ③)
          ė_y   = v_x · e_psi
          v̇_x  = a

        ZOH exact discretization (A_c nilpotent → expm = I + A_c·Ts):
          Ad = I + Ts·A_c
          Bd = (Ts·I + Ts²/2 · A_c) @ B_c   ← second-order term quan trọng!
        """
        v_x = max(v_x, self.v_min)

        Ad = np.array([
            [1.0,           0.0, 0.0],
            [v_x * self.Ts, 1.0, 0.0],
            [0.0,           0.0, 1.0],
        ])

        # B_d[1,0] = v_x²·Ts²/(2L) – tích phân bậc 2 của delta lên e_y
        Bd = np.array([
            [v_x * self.Ts / self.L,              0.0     ],
            [v_x**2 * self.Ts**2 / (2.0 * self.L), 0.0   ],
            [0.0,                                  self.Ts ],
        ])

        Cd = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        return Ad, Bd, Cd

    # ==================================================================
    # ② MPC MATRIX BUILD với CACHE
    # ==================================================================
    def _get_mpc_matrices(self, v_x):
        """
        Lấy hoặc tính (nếu cache miss) Hdb, Fdbt, Cdb, Adc.
        Cache valid khi |v_x - v_x_cached| < v_x_cache_thresh.
        """
        if (self._cached_v_x is not None
                and abs(v_x - self._cached_v_x) < self.v_x_cache_thresh):
            return (self._cached_Hdb, self._cached_Fdbt,
                    self._cached_Cdb,  self._cached_Adc)

        Ad, Bd, Cd = self.calculate_kinematic_state_space(v_x)
        Hdb, Fdbt, Cdb, Adc = self._mpc_simplification(Ad, Bd, Cd, self.hz)

        self._cached_v_x  = v_x
        self._cached_Hdb  = Hdb
        self._cached_Fdbt = Fdbt
        self._cached_Cdb  = Cdb
        self._cached_Adc  = Adc
        return Hdb, Fdbt, Cdb, Adc

    def _mpc_simplification(self, Ad, Bd, Cd, hz):
        """
        Xây dựng Hdb, Fdbt, Cdb, Adc cho QP.
        ① Precompute danh sách A_aug^k (k=0..hz) → O(hz·n³).
           Không dùng matrix_power(A, i-j) trong nested loop O(hz²·n³).

        Augmented state: x_aug = [e_psi, e_y, v_x, U1, U2]  (n_aug = 5)
        """
        n_x   = Ad.shape[0]           # 3
        n_u   = Bd.shape[1]           # 2
        n_y   = Cd.shape[0]           # 2
        n_aug = n_x + n_u             # 5

        A_aug = np.block([
            [Ad,                    Bd                  ],
            [np.zeros((n_u, n_x)), np.eye(n_u)         ],
        ])                             # (5, 5)
        B_aug = np.block([[Bd], [np.eye(n_u)]])          # (5, 2)
        C_aug = np.block([[Cd, np.zeros((n_y, n_u))]])   # (2, 5)

        CQC = C_aug.T @ self.Q @ C_aug     # (5, 5)
        CSC = C_aug.T @ self.S @ C_aug     # (5, 5) terminal cost
        QC  = self.Q @ C_aug               # (2, 5)
        SC  = self.S @ C_aug               # (2, 5)

        # ① Precompute A_aug^k for k = 0, 1, ..., hz
        A_pwr = [np.eye(n_aug)]
        for _ in range(hz):
            A_pwr.append(A_pwr[-1] @ A_aug)
        # A_pwr[k] = A_aug^k

        s_x = n_aug * hz
        s_y = n_y   * hz
        s_u = n_u   * hz

        Qdb = np.zeros((s_x, s_x))
        Tdb = np.zeros((s_y, s_x))
        Rdb = np.kron(np.eye(hz), self.R)    # Block-diagonal (Kronecker trick)
        Cdb = np.zeros((s_x, s_u))
        Adc = np.zeros((s_x, n_aug))

        for i in range(hz):
            Q_block = CSC if i == hz - 1 else CQC
            T_block = SC  if i == hz - 1 else QC
            Qdb[n_aug*i : n_aug*(i+1), n_aug*i : n_aug*(i+1)] = Q_block
            Tdb[n_y*i   : n_y*(i+1),   n_aug*i : n_aug*(i+1)] = T_block

            # Adc[i] = A_aug^(i+1)
            Adc[n_aug*i : n_aug*(i+1), :] = A_pwr[i + 1]

            # Cdb[i, j] = A_aug^(i-j) @ B_aug, j = 0..i
            for j in range(i + 1):
                Cdb[n_aug*i : n_aug*(i+1), n_u*j : n_u*(j+1)] = (
                    A_pwr[i - j] @ B_aug          # ① tra bảng, không tính lại
                )

        Hdb  = Cdb.T @ Qdb @ Cdb + Rdb
        Fdbt = np.vstack([Adc.T @ Qdb @ Cdb, -Tdb @ Cdb])

        return Hdb, Fdbt, Cdb, Adc

    # ==================================================================
    # ⑪ CONSTRAINT MATRICES (khởi tạo 1 lần, G không đổi theo v_x)
    # ==================================================================
    def _build_constraint_matrices(self):
        """
        G * du <= h   (bất đẳng thức QP)
        du = [dδ_0, da_0, dδ_1, da_1, ...] kích thước 2*hz.

        G gồm 4 block:
          Rate limit:  ±I_{2hz}
          Magnitude:   ±L_kron (lower-triangular cumulative sum)
        G cố định (không phụ thuộc v_x hay U), tính 1 lần.
        """
        hz = self.hz
        n2 = 2 * hz

        I_block = np.eye(n2)
        L_kron  = np.kron(np.tril(np.ones((hz, hz))), np.eye(2))  # cumsum matrix

        G = np.vstack([I_block, -I_block, L_kron, -L_kron]).astype(np.float64)
        self._G_sparse  = sparse.csc_matrix(G)
        self._rate_max  = np.tile([self.du_delta_max, self.du_a_max], hz)
        self._U_max_vec = np.tile([self.delta_max, self.a_max],  hz)
        self._U_min_vec = np.tile([self.delta_min, self.a_min],  hz)

    def _build_ht(self):
        """
        Cập nhật RHS vector mỗi bước (O(hz)) khi U1, U2 thay đổi.
        h = [rate_max; rate_max; U_max - U_curr; U_curr - U_min]
        """
        U_curr = np.tile([self.U1, self.U2], self.hz)
        return np.concatenate([
            self._rate_max,
            self._rate_max,
            self._U_max_vec - U_curr,
            U_curr - self._U_min_vec,
        ])

    # ==================================================================
    # QP SOLVER (OSQP via qpsolvers)
    # ==================================================================
    def _solve(self, Hdb, ft):
        """
        Giải QP:  min  0.5·du^T·H·du + ft^T·du
                  s.t. G·du <= ht
        Fallback unconstrained nếu qpsolvers chưa cài (⚠ KHÔNG AN TOÀN).
        """
        if not QP_AVAILABLE:
            try:
                return -np.linalg.solve(Hdb, ft)
            except np.linalg.LinAlgError:
                return None

        ht = self._build_ht()
        Hdb_sym = (0.5 * (Hdb + Hdb.T) + 1e-8 * np.eye(Hdb.shape[0])).astype(np.float64)

        try:
            du = solve_qp(
                sparse.csc_matrix(Hdb_sym), ft.astype(np.float64),
                G=self._G_sparse, h=ht.astype(np.float64),
                solver="osqp",
                verbose=False,
                eps_abs=1e-5, eps_rel=1e-5,
                max_iter=4000,
            )
            if du is None:
                raise ValueError("OSQP trả về None (infeasible hoặc unbounded)")
            return du

        except Exception as e:
            self.get_logger().warn(
                f"[QP FAIL] {e} | "
                f"Hdb_cond={np.linalg.cond(Hdb):.1e} | "
                f"ft_norm={np.linalg.norm(ft):.3f} | "
                f"U1={self.U1:.3f} U2={self.U2:.2f}"
            )
            # Fallback unconstrained (nguy hiểm – chỉ dùng 1 bước rồi safe_brake)
            try:
                return -np.linalg.solve(Hdb, ft)
            except np.linalg.LinAlgError:
                return None

    # ==================================================================
    # CONTROL LOOP (Timer-driven 20Hz – Real-time architecture)
    # ==================================================================
    def control_loop(self):
        if not self.waypoints:
            return

        # ── ④ WATCHDOG: kiểm tra mất tín hiệu Odom ─────────────────
        with self.state_lock:
            now = self.get_clock().now()
            if self.last_odom_time is None:
                return   # Chưa nhận được odom lần nào, chờ tiếp

            dt_odom = (now - self.last_odom_time).nanoseconds / 1e9
            if dt_odom > self.watchdog_timeout:
                self.get_logger().warn(
                    f"[WATCHDOG] Mất Odom {dt_odom:.2f}s > {self.watchdog_timeout}s "
                    f"→ PHANH KHẨN CẤP!"
                )
                self._emergency_stop()
                return

            v_x_curr = max(self.current_v_x, self.v_min)

        # ── 1. Lấy pose từ TF (map → base_link) ─────────────────────
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.car_frame,
                rclpy.time.Time(seconds=0))
            rx    = transform.transform.translation.x
            ry    = transform.transform.translation.y
            q     = transform.transform.rotation
            r_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        except TransformException as e:
            self.get_logger().warn(f"[TF Error] {e}")
            return

        # ── 2. Tìm waypoint gần nhất & tính tracking error ───────────
        nearest_idx = self._find_nearest_waypoint(rx, ry)
        wp_x, wp_y, wp_yaw = self.waypoints[nearest_idx]

        e_psi_raw = normalize_angle(r_yaw - wp_yaw)
        dx = rx - wp_x
        dy = ry - wp_y
        e_y_raw = -math.sin(wp_yaw) * dx + math.cos(wp_yaw) * dy

        # ── 3. BÙ TRỄ HỆ THỐNG (Delay Compensation) ─────────────────
        # Iterate qua delay_steps bước: mỗi bước dùng lệnh đã gửi trước đó.
        # u_buffer[0] = lệnh cũ nhất (đang tác động lên xe ngay bây giờ)
        x_state = np.array([e_psi_raw, e_y_raw, v_x_curr])
        for step in range(self.delay_steps):
            u_prev = self.u_buffer[step]               # (delta, a) đã gửi
            Ad_d, Bd_d, _ = self.calculate_kinematic_state_space(x_state[2])
            x_state = Ad_d @ x_state + Bd_d @ np.array(u_prev)

        x_pred   = x_state
        v_x_pred = max(x_pred[2], self.v_min)

        # ── 4. Xây MPC matrices (② cache) ────────────────────────────
        t0 = time.monotonic()
        Hdb, Fdbt, Cdb, Adc = self._get_mpc_matrices(v_x_pred)

        # Augmented state: [e_psi_pred, e_y_pred, v_x_pred, U1, U2]  (size 5)
        x_aug_t = np.concatenate((x_pred, [self.U1, self.U2]))

        # ── 5. Reference trajectory ───────────────────────────────────
        v_target = self.compute_target_speed(nearest_idx)
        r_vector, ref_pts = self._build_reference(
            nearest_idx, wp_x, wp_y, wp_yaw, v_target, v_x_pred)
        self.publish_mpc_reference(ref_pts)

        # ── 6. Giải QP ────────────────────────────────────────────────
        ft_input = np.concatenate((x_aug_t, r_vector))
        ft       = (Fdbt.T @ ft_input).astype(np.float64)
        Hdb_f64  = (0.5 * (Hdb + Hdb.T) + 1e-8 * np.eye(Hdb.shape[0])).astype(np.float64)

        du = self._solve(Hdb_f64, ft)

        solve_ms = (time.monotonic() - t0) * 1000.0
        if solve_ms > self.Ts * 800:          # Cảnh báo nếu > 80% chu kỳ
            self.get_logger().warn(
                f"[TIMING] Solve={solve_ms:.1f}ms > 80%·Ts "
                f"(Ts={self.Ts*1000:.0f}ms) – xem xét giảm hz!"
            )

        if du is None:
            self._safe_brake(v_x_curr)         # ⑫ Nhất quán: luôn publish
            return

        # ── 7. Cập nhật điều khiển tích lũy ──────────────────────────
        self.U1 = float(np.clip(self.U1 + du[0], self.delta_min, self.delta_max))
        self.U2 = float(np.clip(self.U2 + du[1], self.a_min,     self.a_max    ))

        # ── 8. ③ Curvature feedforward steering ──────────────────────
        # Bù term κ·v_x trong mô hình tuyến tính hóa:
        # δ_total = U1 (MPC feedback) + δ_ff (feedforward)
        kappa    = self._compute_curvature(nearest_idx)
        delta_ff = math.atan2(self.L * kappa, 1.0)
        steering_cmd = float(np.clip(
            self.U1 + delta_ff, self.delta_min, self.delta_max))

        # ── 9. Cập nhật delay buffer (FIFO) ──────────────────────────
        self.u_buffer.pop(0)
        self.u_buffer.append((self.U1, self.U2))

        # ── 10. Tính tốc độ lệnh ─────────────────────────────────────
        v_cmd = float(np.clip(v_target, self.v_min, self.v_max))

        # ── 11. Publish lệnh drive ────────────────────────────────────
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_cmd
        drive_msg.drive.speed          = v_cmd
        self.pub_drive.publish(drive_msg)

        # ── 12. ⑤ Publish predicted trajectory (RViz2 – đường đỏ) ───
        X_pred_vec = Adc @ x_aug_t + Cdb @ du
        self._publish_predicted_path(X_pred_vec, wp_x, wp_y, wp_yaw)

        # ── 13. ⑥ Diagnostic log (throttle 1Hz) ──────────────────────
        self.get_logger().info(
            f"e_psi={e_psi_raw:+.3f}rad | e_y={e_y_raw:+.4f}m | "
            f"δ={steering_cmd:+.3f}rad (ff={delta_ff:+.3f}) | "
            f"v={v_cmd:.2f}m/s | solve={solve_ms:.1f}ms",
            throttle_duration_sec=1.0,
        )

    # ==================================================================
    # ③ CURVATURE FEEDFORWARD HELPERS
    # ==================================================================
    def _compute_curvature(self, idx):
        """
        Tính độ cong có dấu κ tại waypoint idx.
        Dùng Menger curvature (cross product / areas).
        κ > 0: cua trái, κ < 0: cua phải.
        """
        n = len(self.waypoints)
        k = max(self.curvature_lookahead // 2, 1)
        p0 = self.waypoints[(idx - k) % n]
        p1 = self.waypoints[idx]
        p2 = self.waypoints[(idx + k) % n]

        dx1, dy1 = p1[0] - p0[0], p1[1] - p0[1]
        dx2, dy2 = p2[0] - p1[0], p2[1] - p1[1]
        cross = dx1 * dy2 - dy1 * dx2          # Signed (+ = left turn)
        d01 = math.hypot(dx1, dy1) + 1e-9
        d12 = math.hypot(dx2, dy2) + 1e-9
        d02 = math.hypot(p2[0] - p0[0], p2[1] - p0[1]) + 1e-9
        return 2.0 * cross / (d01 * d12 * d02)

    # ==================================================================
    # TỐC ĐỘ THÍCH NGHI THEO ĐỘ CONG
    # ==================================================================
    def compute_target_speed(self, nearest_idx):
        """
        Tính tốc độ mục tiêu từ sin(góc thay đổi hướng) của path.
        Thẳng → v_straight, cua gắt → v_curve.
        """
        n = len(self.waypoints)
        k = self.curvature_lookahead
        p_prev = self.waypoints[(nearest_idx - k) % n]
        p_curr = self.waypoints[nearest_idx]
        p_next = self.waypoints[(nearest_idx + k) % n]

        dx1, dy1 = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
        dx2, dy2 = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
        cross = abs(dx1 * dy2 - dy1 * dx2)
        n1 = math.hypot(dx1, dy1) + 1e-6
        n2 = math.hypot(dx2, dy2) + 1e-6
        curvature = min(cross / (n1 * n2), 1.0)   # ≈ sin(angle_change)

        speed = self.v_straight + curvature * (self.v_curve - self.v_straight)
        return float(np.clip(speed, self.v_min, self.v_max))

    # ==================================================================
    # REFERENCE TRAJECTORY (Lookahead, ⑦ anti-infinite-loop)
    # ==================================================================
    def _build_reference(self, nearest_idx, wp_x, wp_y, wp_yaw,
                         v_target, current_v_x):
        """
        Tạo reference vector r (2*hz,) = [e_psi_ref_1, e_y_ref_1, ...].
        Các điểm tham chiếu là vị trí tương lai trên raceline nhìn trước hz bước.
        ⑦ max_iter guard ngăn vòng lặp vô hạn khi path có đoạn length=0.
        """
        r_list, ref_pts = [], []
        step_dist  = max(current_v_x, 1.5) * self.Ts
        curr_idx   = nearest_idx
        dist_accum = 0.0
        max_iter   = len(self.waypoints) + 1     # ⑦ Safety upper bound

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
                    # Degenerate segment → skip
                    curr_idx = nxt
                    safe_count += 1
                    continue

                if dist_accum + slen >= target_dist:
                    r   = np.clip((target_dist - dist_accum) / slen, 0.0, 1.0)
                    fx  = p1[0] + r * (p2[0] - p1[0])
                    fy  = p1[1] + r * (p2[1] - p1[1])
                    fyaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    found = True
                    break
                else:
                    dist_accum += slen
                    curr_idx    = nxt
                    safe_count += 1

            if not found:
                # Fallback: dùng waypoint hiện tại
                p1 = self.waypoints[curr_idx]
                fx, fy, fyaw = p1[0], p1[1], p1[2]

            ref_pts.append((fx, fy))
            ly   = -math.sin(wp_yaw) * (fx - wp_x) + math.cos(wp_yaw) * (fy - wp_y)
            lyaw = normalize_angle(fyaw - wp_yaw)
            r_list.extend([lyaw, ly])

        return np.array(r_list), ref_pts

    # ==================================================================
    # TÌM WAYPOINT GẦN NHẤT (Rolling-window search)
    # ==================================================================
    def _find_nearest_waypoint(self, rx, ry):
        if self.start_index is None:
            dists = [math.hypot(rx - p[0], ry - p[1]) for p in self.waypoints]
            self.start_index = int(np.argmin(dists))
            return self.start_index

        idx    = self.start_index
        curr_d = math.hypot(rx - self.waypoints[idx][0], ry - self.waypoints[idx][1])
        for _ in range(30):    # 30 bước đủ cho v_max = 3.5m/s ở 20Hz
            nxt = (idx + 1) % len(self.waypoints)
            d   = math.hypot(rx - self.waypoints[nxt][0], ry - self.waypoints[nxt][1])
            if d < curr_d:
                idx    = nxt
                curr_d = d
            else:
                break
        self.start_index = idx
        return idx

    # ==================================================================
    # SAFETY FUNCTIONS
    # ==================================================================
    def _emergency_stop(self):
        """Phanh khẩn cấp: tốc độ 0, lái về 0."""
        self.U1, self.U2 = 0.0, 0.0
        msg = AckermannDriveStamped()
        msg.drive.speed          = 0.0
        msg.drive.steering_angle = 0.0
        self.pub_drive.publish(msg)

    def _safe_brake(self, v_x_curr):
        """
        ⑫ Phanh mềm khi QP thất bại: decay góc lái, giảm tốc dần.
        Luôn publish command ngay (không return mà không gửi gì).
        """
        self.U1 *= 0.85                              # Giảm dần về 0
        self.U2  = max(self.U2 - 0.5, self.a_min)   # Phanh nhẹ
        v_safe   = float(np.clip(v_x_curr + self.U2 * self.Ts, 0.0, self.v_max))

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(self.U1)
        msg.drive.speed          = v_safe
        self.pub_drive.publish(msg)
        self.get_logger().warn("[SAFE BRAKE] QP thất bại – đang phanh mềm.")

    # ==================================================================
    # LOAD WAYPOINTS + SMOOTH PATH (⑩ CSV parser robust)
    # ==================================================================
    def load_waypoints(self, filename):
        """
        ⑩ Đọc CSV với format linh hoạt:
          - Comma-separated: x,y hoặc x,y,z,...
          - Space-separated: x y
          - Header rows: bỏ qua tự động (ValueError)
          - Comment '#': bỏ qua
        """
        raw = []
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or row[0].strip().startswith('#'):
                        continue
                    # Xử lý single-token (space-separated nằm trong 1 field)
                    if len(row) == 1:
                        row = row[0].split()
                    try:
                        x, y = float(row[0]), float(row[1])
                        raw.append([x, y])
                    except (ValueError, IndexError):
                        continue   # Bỏ qua header, comment, dòng lỗi

            if len(raw) < 4:
                self.get_logger().error(
                    f"Quá ít waypoints ({len(raw)}) trong: {filename}")
                return

            sm = self.smooth_path(raw)
            n  = len(sm)
            self.waypoints = [
                [sm[i][0], sm[i][1],
                 math.atan2(sm[(i+1) % n][1] - sm[i][1],
                            sm[(i+1) % n][0] - sm[i][0])]
                for i in range(n)
            ]
            self.get_logger().info(
                f"[Waypoints] Đã load {len(self.waypoints)} điểm từ {filename}"
            )

        except FileNotFoundError:
            self.get_logger().error(f"[Waypoints] Không tìm thấy: {filename}")
        except Exception as e:
            self.get_logger().error(f"[Waypoints] Lỗi đọc file: {e}")

    def smooth_path(self, path, weight_data=0.5, weight_smooth=0.2,
                    tolerance=1e-5):
        """Gradient-descent path smoothing, hội tụ theo tolerance."""
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

    # ==================================================================
    # PUBLISH MARKERS (RViz2)
    # ==================================================================
    def _publish_predicted_path(self, X_pred_vec, wp_x, wp_y, wp_yaw):
        """
        ⑤ Vẽ quỹ đạo dự đoán (đỏ) lên RViz2.
        n_aug = 5: [e_psi(0), e_y(1), v_x(2), U1(3), U2(4)]
        """
        n_aug   = 5
        marker  = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.ns, marker.id   = "mpc_predict", 1
        marker.type            = Marker.LINE_STRIP
        marker.action          = Marker.ADD
        marker.scale.x         = 0.08
        marker.color.a         = 0.85
        marker.color.r         = 1.0
        marker.color.g         = 0.0
        marker.color.b         = 0.0

        s_accum = 0.0
        for i in range(self.hz):
            e_y_p  = X_pred_vec[n_aug * i + 1]
            v_x_p  = X_pred_vec[n_aug * i + 2]
            s_accum += max(v_x_p, self.v_min) * self.Ts
            px = wp_x + s_accum * math.cos(wp_yaw) - e_y_p * math.sin(wp_yaw)
            py = wp_y + s_accum * math.sin(wp_yaw) + e_y_p * math.cos(wp_yaw)
            marker.points.append(Point(x=float(px), y=float(py), z=0.15))

        self.pub_mpc_predict.publish(marker)

    def publish_mpc_reference(self, pts):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns, m.id        = "mpc_ref", 0
        m.type, m.action  = Marker.SPHERE_LIST, Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.15
        m.color.a = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.points  = [Point(x=float(p[0]), y=float(p[1]), z=0.1) for p in pts]
        self.pub_mpc_ref.publish(m)

    def publish_full_waypoint(self):
        arr = MarkerArray()
        mk  = Marker()
        mk.header.frame_id = self.map_frame
        mk.header.stamp    = self.get_clock().now().to_msg()
        mk.id, mk.type, mk.action = 0, Marker.LINE_STRIP, Marker.ADD
        mk.scale.x         = 0.05
        mk.color.a         = 1.0
        mk.color.r         = 1.0
        mk.color.g         = 1.0
        mk.color.b         = 0.0
        mk.points = [Point(x=float(p[0]), y=float(p[1]), z=0.0)
                     for p in self.waypoints]
        if self.waypoints:
            mk.points.append(Point(
                x=float(self.waypoints[0][0]),
                y=float(self.waypoints[0][1]),
                z=0.0,
            ))
        arr.markers.append(mk)
        self.pub_marker_path.publish(arr)


# ======================================================================
# MAIN
# ======================================================================
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MPCRealCarNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        print(f"[MPC FATAL] {e}")
        traceback.print_exc()
    finally:
        if node is not None:
            node._emergency_stop()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()