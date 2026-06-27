#!/usr/bin/env python3
"""
MPPI Obstacle Avoidance Controller — F1TENTH ROS 2
Dựa trên lý thuyết: Williams et al. 2018, "Information-Theoretic MPC"
Tham khảo implementation: MizuhoAOKI/python_simple_mppi, UM-ARM-Lab/pytorch_mppi

Các lỗi đã sửa so với bản gốc:
  BUG-1: Tọa độ vật cản (obstacles) giờ được chuyển sang map frame trong lidar_callback
         qua TF, đảm bảo compute_cost so sánh pts (map) với obstacles (map) nhất quán.
  BUG-2: Receding horizon shift — lưu last_steer TRƯỚC khi shift, không bị đọc nhầm vị trí.
  BUG-3: Visualize quỹ đạo danh nghĩa (rollout từ nominal_control) thay vì mẫu argmax.
  BUG-4: Dùng effective_noise = perturbed_clipped - nominal cho weight update,
         đảm bảo rollout và update nhất quán khi có clipping.
Tối ưu hóa:
  OPT-1: Track cost chỉ dùng wp_window waypoints gần nhất (giảm từ ~2 GB xuống ~12 MB).
  OPT-2: np.isfinite lọc NaN/Inf từ LiDAR trước khi xử lý.
  OPT-3: control smoothness gộp 2 channel vào 1 dòng sum để giảm tạo mảng tạm.
  OPT-4: Decoupled control_loop (20Hz) song song đa luồng (MultiThreadedExecutor)
         giảm tải CPU và loại bỏ sensor lag.
  OPT-5: Tính track cost bằng np.einsum tránh phép tính sqrt thừa trên mảng lớn.
"""

import csv
import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class MPPIController(Node):
    def __init__(self):
        super().__init__("mppi_controller_node")

        # ── Thông số xe ──────────────────────────────────────────────
        self.L  = 0.33    # Chiều dài trục cơ sở F1TENTH (m)
        self.dt = 0.05    # Chu kỳ lấy mẫu (20 Hz)

        # ── Thông số MPPI ─────────────────────────────────────────────
        self.horizon     = 25     # Số bước nhìn trước (1.25 s)
        self.num_samples = 500    # Số quỹ đạo mẫu ngẫu nhiên

        # Độ lệch chuẩn nhiễu Gauss: [tốc độ m/s, góc lái rad]
        # noise steer lớn hơn (0.15 -> 0.25) để né tránh chướng ngại vật khẩn cấp tốt hơn
        # tăng noise speed để tối ưu hóa việc tăng/giảm tốc ở tốc độ cao
        self.noise_sigma = np.array([1.0, 0.25])

        # Temperature λ: tương thích với cost scale sau chuẩn hóa
        self.lambda_ = 50.0

        # ── Giới hạn cơ giới ─────────────────────────────────────────
        self.max_speed = 6.0    # Tăng giới hạn tốc độ tối đa lên 6.0 m/s để xe có thể chạy nhanh hơn
        self.min_speed = 0.0
        self.max_steer = 0.35   # ~20 độ

        self.w_track    = 80.0  # Bám đường raceline chặt (tăng từ 40 để xe bám sát vạch đường khi đi cực nhanh)
        self.w_progress = 1.5   # Tiến dọc đường đua (giảm để không lấn át cost tránh vật cản/bám cua)
        self.w_control  = 1.5   # Làm mịn lệnh điều khiển
        self.w_obstacle = 100.0 # Tránh vật cản (va chạm chuẩn hóa)
        self.w_speed    = 5.0   # Bám vận tốc mục tiêu (giảm từ 15 để ưu tiên cho sự an toàn và bám đường lên trước)
        self.w_heading  = 15.0  # Bám hướng tiếp tuyến

        # Bán kính an toàn của xe (m)
        self.robot_radius   = 0.35
        self.danger_radius  = 1.10  # Tăng lên 1.10m để phát hiện và phản ứng sớm hơn với chướng ngại vật

        # Tốc độ mục tiêu lớn nhất trên đường thẳng (m/s)
        self.target_speed = 5.0  # Tăng tốc độ mục tiêu trên đường thẳng lên 5.0 m/s

        # ── Tham số curvature-based speed profiling ─────────────────
        self.min_speed_curve = 1.8     # Tăng tốc độ tối thiểu khi vào cua gắt lên 1.8 m/s để duy trì động năng
        self.curve_threshold = 0.35    # Ngưỡng độ cong (rad/m) bắt đầu giảm tốc (tăng từ 0.28 lên 0.35 để cho phép cua nhanh hơn)
        self.lookahead_wps   = 15      # Tăng số lượng waypoints nhìn trước lên 15 để phanh sớm trước cua từ tốc độ cao

        # Cửa sổ waypoint cục bộ
        self.wp_window = 50  # Tăng để có đủ waypoints nhìn trước

        # ── Chuỗi điều khiển danh nghĩa U: (T, 2) → [speed, steer] ──
        self.nominal_control = np.zeros((self.horizon, 2))
        self.nominal_control[:, 0] = self.target_speed

        # Vật cản trong MAP frame (cập nhật từ lidar_callback qua TF)
        self.map_obstacles = np.zeros((0, 2))
        self.obstacle_stamp = None

        # Vật cản ảo được thêm qua click point trên RViz
        self.virtual_obstacles = []

        # ── Hysteresis State cho việc đi lùi ──────────────────────────
        self.is_reversing   = False
        self.forward_min_obs_dist = 999.0  # Khoảng cách tới vật cản phía trước mặt
        self.last_best_obs_cost = 0.0

        # Vận tốc dọc hiện tại (cập nhật từ odom_callback)
        self.v_cur = 0.0

        # RNG riêng cho MPPI (nhanh và an toàn đa luồng hơn)
        self.rng = np.random.default_rng()

        # ── TF ───────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.car_frame   = "ego_racecar/base_link"
        self.map_frame   = "map"

        # ── Callback groups ───────────────────────────────────────────
        sensor_grp  = ReentrantCallbackGroup()
        control_grp = MutuallyExclusiveCallbackGroup()

        # ── ROS 2 pub/sub ─────────────────────────────────────────────
        self.sub_odom  = self.create_subscription(Odometry,  "ego_racecar/odom", self.odom_callback,  10, callback_group=sensor_grp)
        self.sub_laser = self.create_subscription(LaserScan, "/scan",             self.lidar_callback, 10, callback_group=sensor_grp)
        self.sub_clicked_point = self.create_subscription(PointStamped, "/clicked_point", self.clicked_point_callback, 10, callback_group=sensor_grp)
        self.sub_clear_virtual = self.create_subscription(Empty, "/clear_virtual_obstacles", self.clear_virtual_obstacles_callback, 10, callback_group=sensor_grp)

        self.pub_drive     = self.create_publisher(AckermannDriveStamped, "/drive",                 10)
        self.pub_best_traj = self.create_publisher(Marker,                "/mppi_best_trajectory",  10)
        self.pub_waypoints = self.create_publisher(MarkerArray,           "/publish_full_waypoint", 10)
        self.pub_virtual_obs = self.create_publisher(Marker,              "/virtual_obstacles_marker", 10)

        # ── Vòng lặp điều khiển cố định 20 Hz (tách độc lập đa luồng) ──
        self.control_timer = self.create_timer(self.dt, self.control_loop, callback_group=control_grp)

        # ── Nạp waypoints ─────────────────────────────────────────────
        self.waypoints          = np.zeros((0, 2))
        self.waypoint_headings  = np.zeros(0)   # Góc tiếp tuyến tại mỗi wp
        csv_path = (
            "/sim_ws/install/waypoint/share/waypoint"
            "/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        )
        self._load_waypoints(csv_path)
        self._publish_waypoints()

        # ── Thống kê cost để log ─────────────────────────────────────
        self.log_counter    = 0
        self._diag_done     = False   # Flag: chỉ in diagnostic 1 lần
        self._dbg_track     = 0.0
        self._dbg_progress  = 0.0
        self._dbg_speed     = 0.0
        self._dbg_heading   = 0.0
        self._dbg_obs       = 0.0
        self._dbg_n_obs     = 0
        self.get_logger().info("MPPI Controller started.")

    # ─────────────────────────────────────────────────────────────────
    # Tiện ích
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _quat_to_yaw(q) -> float:
        """Quaternion (x,y,z,w) → góc yaw (rad)."""
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return float(np.arctan2(siny, cosy))

    def _load_waypoints(self, csv_path: str) -> None:
        pts = []
        try:
            with open(csv_path, "r") as f:
                for row in csv.reader(f):
                    if not row or row[0].strip().startswith("#"):
                        continue
                    try:
                        pts.append([float(row[0]), float(row[1])])
                    except (ValueError, IndexError):
                        continue  # bỏ qua hàng tiêu đề hoặc dòng lỗi
            self.waypoints = np.array(pts) if pts else np.zeros((0, 2))
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")

            # Tính trước góc hướng (heading) tiếp tuyến và độ cong (curvature) tại từng waypoint
            # để bổ sung Heading Alignment Cost và Curvature Speed Profiling
            if self.waypoints.shape[0] > 1:
                diffs = np.diff(self.waypoints, axis=0)         # (W-1, 2)
                last  = self.waypoints[0] - self.waypoints[-1]  # wrap-around
                diffs = np.vstack([diffs, last])                # (W, 2)
                self.waypoint_headings = np.arctan2(diffs[:, 1], diffs[:, 0])
                self.get_logger().info("Waypoint headings computed OK.")

                # Curvature = |d_theta| / ds
                ds = np.linalg.norm(diffs, axis=1)
                ds[ds < 1e-3] = 1.0  # tránh chia cho 0
                
                # Sự thay đổi góc hướng giữa các waypoint liên tiếp (wrapped to [-pi, pi])
                hdg_diff = np.diff(self.waypoint_headings)
                last_hdg_diff = self.waypoint_headings[0] - self.waypoint_headings[-1]
                hdg_diff = np.append(hdg_diff, last_hdg_diff)
                hdg_diff = (hdg_diff + np.pi) % (2.0 * np.pi) - np.pi
                
                self.waypoint_curvatures = np.abs(hdg_diff) / ds
                self.get_logger().info("Waypoint curvatures computed OK.")
            else:
                self.waypoint_headings = np.zeros(self.waypoints.shape[0])
                self.waypoint_curvatures = np.zeros(self.waypoints.shape[0])
        except Exception as e:
            self.get_logger().error(f"Cannot load CSV: {e}")

    # ─────────────────────────────────────────────────────────────────
    # LiDAR callback — chuyển obstacles sang MAP frame ngay lúc nhận
    # ─────────────────────────────────────────────────────────────────

    def lidar_callback(self, msg: LaserScan) -> None:
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # Lọc nhiễu: loại bỏ điểm quét quá gần, quá xa, NaN, Inf
        valid = np.isfinite(ranges) & (ranges > 0.15) & (ranges < 4.5)
        r   = ranges[valid]
        phi = angles[valid]

        # Khoảng cách tới vật cản phía trước mặt (cone -40 đến +40 độ) để kích hoạt lùi
        forward_mask = np.abs(phi) < np.radians(40)
        if forward_mask.any():
            self.forward_min_obs_dist = float(np.min(r[forward_mask]))
        else:
            self.forward_min_obs_dist = 999.0

        # ── SUBSAMPLE ưu tiên khoảng cách gần (Distance-Priority Subsampling) ──
        # Giữ lại các điểm cực gần (< 2m) để tránh lọt chướng ngại vật nhỏ, subsample các điểm ở xa
        NEAR_THRESHOLD = 2.0
        MAX_OBS = 80

        near_mask = r < NEAR_THRESHOLD
        r_near, phi_near = r[near_mask], phi[near_mask]
        r_far, phi_far = r[~near_mask], phi[~near_mask]

        remaining = max(0, MAX_OBS - len(r_near))
        if len(r_far) > remaining > 0:
            step = len(r_far) // remaining
            r_far = r_far[::step][:remaining]
            phi_far = phi_far[::step][:remaining]
        elif len(r_near) > MAX_OBS:
            step = len(r_near) // MAX_OBS
            r_near = r_near[::step][:MAX_OBS]
            phi_near = phi_near[::step][:MAX_OBS]
            r_far = np.zeros(0)
            phi_far = np.zeros(0)

        r = np.concatenate([r_near, r_far])
        phi = np.concatenate([phi_near, phi_far])

        # Tọa độ trong base_link
        x_car = r * np.cos(phi)
        y_car = r * np.sin(phi)

        # Chuyển sang MAP frame qua TF (FIX BUG-1: nhất quán khung tọa độ)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02),
            )
            tx  = tf.transform.translation.x
            ty  = tf.transform.translation.y
            q   = tf.transform.rotation
            yaw = np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)

            x_map = cos_y * x_car - sin_y * y_car + tx
            y_map = sin_y * x_car + cos_y * y_car + ty
            self.map_obstacles = np.stack([x_map, y_map], axis=1)
            # Dùng node clock (không phải msg stamp) để TTL so sánh cùng nguồn thời gian với control_loop
            self.obstacle_stamp = self.get_clock().now()
        except Exception:
            # TF chưa sẵn sàng: giữ nguyên map_obstacles cũ (fail-safe)
            pass

    # ─────────────────────────────────────────────────────────────────
    # Mô hình động học xe đạp vector hóa
    # states:   (N, 3)  [x, y, theta]
    # controls: (N, 2)  [v, delta]
    # ─────────────────────────────────────────────────────────────────

    def _step(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        x, y, th = states[:, 0], states[:, 1], states[:, 2]
        v, d     = controls[:, 0], controls[:, 1]
        dt       = self.dt

        # Mô phỏng trượt (understeer) ở tốc độ cao: giảm hiệu quả đánh lái trong mô hình dự báo
        # để bộ điều khiển MPPI tự động tăng góc lái thực tế bù vào.
        slip_factor = 1.0 / (1.0 + v * 0.15)
        d_eff = d * slip_factor

        return np.stack(
            [
                x  + v * np.cos(th) * dt,
                y  + v * np.sin(th) * dt,
                th + (v * np.tan(d_eff) / self.L) * dt,
            ],
            axis=1,
        )

    # ─────────────────────────────────────────────────────────────────
    # Lấy cửa sổ waypoint gần nhất (OPT-1)
    # ─────────────────────────────────────────────────────────────────

    def _local_waypoints(self, x: float, y: float):
        """Trả về (wps, headings, global_indices)."""
        if self.waypoints.shape[0] == 0:
            return self.waypoints, np.zeros(0), np.zeros(0, dtype=int)
        dx      = self.waypoints[:, 0] - x
        dy      = self.waypoints[:, 1] - y
        nearest = int(np.argmin(dx * dx + dy * dy))
        n       = len(self.waypoints)
        # Bắt đầu cửa sổ từ 15 điểm phía sau đến wp_window-15 điểm phía trước để hỗ trợ
        # xe đi lùi hoặc lệch phía sau nearest waypoint mà vẫn tính toán đúng tiến trình/hướng.
        idx     = (np.arange(-15, self.wp_window - 15) + nearest) % n
        return self.waypoints[idx], self.waypoint_headings[idx], idx

    # ─────────────────────────────────────────────────────────────────
    # Hàm chi phí đa mục tiêu (vectorized)
    # state_rollouts:    (N, T+1, 3)
    # perturbed_controls: (N, T, 2)
    # pos:               (x, y) vị trí hiện tại để chọn waypoints
    # ─────────────────────────────────────────────────────────────────

    def _compute_cost(
        self,
        state_rollouts:     np.ndarray,
        perturbed_controls: np.ndarray,
        pos:                tuple,
        theta:              float,
        target_speed:       float,
    ) -> np.ndarray:
        """Hàm chi phí MPPI.
        Tham khảo: Williams 2018, vaithak/f1tenth_shield_mppi, pbarry670/f1tenth-mppi.
        """
        N, T = self.num_samples, self.horizon
        pts = state_rollouts[:, 1:, :2]   # (N, T, 2)
        ths = state_rollouts[:, 1:, 2]    # (N, T) — yaw

        # Lấy waypoints + headings cục bộ
        local_wps, local_hdgs, local_indices = self._local_waypoints(pos[0], pos[1])
        W = local_wps.shape[0]
        if W == 0:
            return np.full(N, 1e6)

        # ── Khoảng cách mỗi rollout đến mỗi waypoint ────────────────
        # OPT-5: Sử dụng np.einsum để tính bình phương khoảng cách, tránh phép tính sqrt thừa trên mảng lớn
        delta_wp  = pts[:, :, None, :] - local_wps[None, None, :, :]  # (N,T,W,2)
        sq_wp     = np.einsum("ntwk,ntwk->ntw", delta_wp, delta_wp)    # (N,T,W)
        min_wi    = np.argmin(sq_wp, axis=2)                           # (N,T)  int
        min_sq    = np.take_along_axis(sq_wp, min_wi[:, :, None], axis=2)[:, :, 0]  # (N,T)

        # ── 1. Cross-track cost (bình phương khoảng cách ngang) ─────
        track_cost = np.sum(min_sq, axis=1) / T                        # (N,) chuẩn hóa

        # ── 2. Progress reward (tiến dọc theo đường đua - BUG-B Modular Progress) ──
        # Tìm index wp gần nhất tại bước t=0 (vị trí hiện tại)
        dx0   = local_wps[:, 0] - pos[0]
        dy0   = local_wps[:, 1] - pos[1]
        idx0  = int(np.argmin(dx0 * dx0 + dy0 * dy0))
        
        # Lấy chỉ số toàn cục (global indices) để tính chênh lệch tiến trình vòng tròn chính xác
        rollout_global_wps = local_indices[min_wi]                     # (N, T)
        nearest_global_idx = local_indices[idx0]                       # int
        num_wps = self.waypoints.shape[0]
        progress_raw = (rollout_global_wps - nearest_global_idx) % num_wps
        
        # Chuyển sang hiệu số có dấu để đi lùi bị tính là tiến trình âm
        backwards_mask = progress_raw > (num_wps // 2)
        progress_raw[backwards_mask] -= num_wps
        
        # Chuẩn hóa về [0, 1], thưởng nếu tiến xa hơn
        progress_mean  = progress_raw.astype(float).mean(axis=1)  # (N,)
        # Đây là REWARD nên cost = -progress, dùng âm để minimize
        progress_cost  = -progress_mean                            # (N,)

        # ── 3. Heading cost (bám hướng tiếp tuyến) ──────────────────
        target_hdgs  = local_hdgs[min_wi]                          # (N,T)
        heading_err  = ths - target_hdgs
        heading_err  = (heading_err + np.pi) % (2.0 * np.pi) - np.pi
        # Không clip heading_err ở [-pi/2, pi/2] nữa nhằm giữ lại độ dốc thông tin toàn dải [-pi, pi],
        # giúp MPPI tự động quay đầu xe lại khi bị xoay ngược 180 độ.
        heading_cost = np.sum(heading_err ** 2, axis=1) / T        # (N,)

        # ── 4. Speed cost ────────────────────────────────────────────
        speed_err  = perturbed_controls[:, :, 0] - target_speed
        speed_cost = np.sum(speed_err ** 2, axis=1) / T            # (N,)

        # ── 5. Control smoothness cost ───────────────────────────────
        ctrl_diff   = np.diff(perturbed_controls, axis=1)
        smooth_cost = np.sum(ctrl_diff ** 2, axis=(1, 2)) / T      # (N,)

        # ── 6. Obstacle cost (chuẩn hóa về [0, 10.0] tránh Softmax Collapse) ──
        obs_cost       = np.zeros(N)
        n_obs_filtered = 0
        
        obstacles_list = []
        if self.map_obstacles.shape[0] > 0:
            obstacles_list.append(self.map_obstacles)
        if len(self.virtual_obstacles) > 0:
            obstacles_list.append(np.array(self.virtual_obstacles))
            
        if len(obstacles_list) > 0:
            all_obstacles = np.vstack(obstacles_list)
            max_travel    = self.max_speed * self.horizon * self.dt
            filter_radius = max_travel + self.danger_radius + 0.2
            dx = all_obstacles[:, 0] - pos[0]
            dy = all_obstacles[:, 1] - pos[1]
            sq_dist        = dx * dx + dy * dy
            close_mask     = sq_dist < (filter_radius ** 2)
            local_obs      = all_obstacles[close_mask]
            n_obs_filtered = int(local_obs.shape[0])

            # Giới hạn tối đa 30 điểm gần nhất
            MAX_OBS_COST = 30
            if n_obs_filtered > MAX_OBS_COST:
                close_dists = np.sqrt(sq_dist[close_mask])
                nearest_idx = np.argsort(close_dists)[:MAX_OBS_COST]
                local_obs   = local_obs[nearest_idx]
                n_obs_filtered = MAX_OBS_COST

            if n_obs_filtered > 0:
                delta    = pts[:, :, None, :] - local_obs[None, None, :, :]  # (N,T,M,2)
                dists    = np.linalg.norm(delta, axis=-1)                     # (N,T,M)
                collision = (dists < self.robot_radius).any(axis=2)           # (N,T)
                col_cost  = collision.sum(axis=1).astype(float)               # (N,)
                
                # Phạt nếu va chạm bất kỳ bước nào trong horizon
                collision_any = collision.any(axis=1).astype(float)          # (N,)
                
                # Lấy khoảng cách vật cản gần nhất tại mỗi bước để tránh loãng chi phí (cost dilution)
                min_dists = np.min(dists, axis=2)                            # (N, T)
                sigma     = (self.danger_radius - self.robot_radius) / 2.0
                gauss     = np.exp(-0.5 * ((min_dists - self.robot_radius) / sigma) ** 2)
                mask      = (min_dists < self.danger_radius) & (min_dists >= self.robot_radius)
                soft_cost = np.sum(gauss * mask, axis=1) / T                 # (N,)
                
                # Chuẩn hóa obs_cost về [0, 10.0] để áp đảo progress reward khi va chạm xảy ra
                obs_cost = np.clip(
                    collision_any * 5.0 +
                    (col_cost / T) * 2.0 +
                    soft_cost * 3.0,
                    0.0, 10.0
                )

        # ── 7. Terminal cost (tập trung tại bước cuối cùng t=T - BUG-F) ──
        final_pts = state_rollouts[:, -1, :2]   # (N, 2)
        
        # Phạt lệch đường cuối horizon
        dx_f = final_pts[:, 0:1] - local_wps[None, :, 0]
        dy_f = final_pts[:, 1:2] - local_wps[None, :, 1]
        dist_f = np.sqrt(dx_f**2 + dy_f**2)
        min_f = dist_f.min(axis=1)             # (N,)
        
        # Phạt va chạm cuối horizon
        terminal_obs = np.zeros(N)
        if n_obs_filtered > 0:
            d_obs_f = np.linalg.norm(
                final_pts[:, None, :] - local_obs[None, :, :], axis=-1
            ).min(axis=1)                      # (N,)
            terminal_obs = (d_obs_f < self.danger_radius).astype(float)
            
        terminal_cost = 3.0 * (self.w_track * min_f**2 + self.w_obstacle * terminal_obs)

        # ── Lưu thống kê log ─────────────────────────────────────────
        self._dbg_track    = float(np.mean(self.w_track    * track_cost))
        # progress: hiện max cho thấy rollout tốt nhất tiến được bao xa
        self._dbg_progress = float(np.max(progress_mean))   # waypoints / horizon
        self._dbg_speed    = float(np.mean(self.w_speed    * speed_cost))
        self._dbg_heading  = float(np.mean(self.w_heading  * heading_cost))
        self._dbg_obs      = float(np.mean(self.w_obstacle * obs_cost))
        self._dbg_n_obs    = n_obs_filtered

        # Lưu lại để control_loop trích xuất của best rollout
        self._current_track_cost = track_cost
        self._current_progress_cost = progress_cost
        self._current_speed_cost = speed_cost
        self._current_heading_cost = heading_cost
        self._current_smooth_cost = smooth_cost
        self._current_obs_cost = obs_cost

        return (
            self.w_track    * track_cost    +
            self.w_progress * progress_cost +  # âm → giảm tổng cost khi tiến xa
            self.w_control  * smooth_cost   +
            self.w_speed    * speed_cost    +
            self.w_heading  * heading_cost  +
            self.w_obstacle * obs_cost      +
            terminal_cost
        )

    # ─────────────────────────────────────────────────────────────────
    # Vòng lặp điều khiển chính (20 Hz)
    # ─────────────────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry) -> None:
        """Callback cực nhẹ: chỉ lưu lại vận tốc dọc hiện tại để tránh lag."""
        self.v_cur = float(msg.twist.twist.linear.x)

    def control_loop(self) -> None:
        """Vòng lặp tính toán MPPI chạy cố định ở 20 Hz trên luồng riêng."""
        start_time = self.get_clock().now()
        self.log_counter += 1

        # 1. Lấy trạng thái hiện tại từ TF (Hệ tọa độ Map) thay vì đọc trực tiếp tin nhắn Odom
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.car_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02),
            )
            x0 = tf.transform.translation.x
            y0 = tf.transform.translation.y
            q  = tf.transform.rotation
            theta0 = self._quat_to_yaw(q)
        except Exception as e:
            self.get_logger().warn(f"Chua co vi tri tren Map qua TF: {str(e)}", throttle_duration_sec=2.0)
            self._publish_drive(0.0, 0.0) # Fail-safe phanh xe khi mất định vị
            return

        v_cur  = self.v_cur
        state  = np.array([x0, y0, theta0])

        if self.waypoints.shape[0] == 0:
            self.get_logger().error("Khong co waypoint — phanh xe khẩn cấp.", throttle_duration_sec=2.0)
            self._publish_drive(0.0, 0.0)
            return

        # Kiểm tra TTL của vật cản (tránh phantom obstacles khi ngắt kết nối LiDAR)
        now_time = self.get_clock().now()
        lidar_lost = False
        if self.obstacle_stamp is not None:
            dt_obs = (now_time - self.obstacle_stamp).nanoseconds / 1e9
            if dt_obs > 0.5:
                self.map_obstacles = np.zeros((0, 2))
                self.forward_min_obs_dist = 999.0
                lidar_lost = True
        else:
            self.map_obstacles = np.zeros((0, 2))
            self.forward_min_obs_dist = 999.0
            lidar_lost = True

        # ── Tính khoảng cách chướng ngại vật gần nhất ────────────────
        min_obs_dist = 999.0
        obstacles_list = []
        if self.map_obstacles.shape[0] > 0:
            obstacles_list.append(self.map_obstacles)
        if len(self.virtual_obstacles) > 0:
            obstacles_list.append(np.array(self.virtual_obstacles))
        if len(obstacles_list) > 0:
            all_obs = np.vstack(obstacles_list)
            obs_dists = np.linalg.norm(all_obs - np.array([x0, y0]), axis=1)
            min_obs_dist = float(np.min(obs_dists))

        # ── Hysteresis State cho việc đi lùi (Tránh dao động tiến/lùi liên tục) ──
        if not self.is_reversing:
            if self.forward_min_obs_dist < 0.8:
                self.is_reversing = True
                self.get_logger().warn(
                    f"[SAFETY] Cận kề va chạm phía trước (forward_obs={self.forward_min_obs_dist:.2f}m < 0.8m) -> Kích hoạt chế độ lùi tự động."
                )
        else:
            if self.forward_min_obs_dist > 1.2:
                self.is_reversing = False
                self.get_logger().info(
                    f"[SAFETY] Đã lùi ra khoảng cách an toàn phía trước (forward_obs={self.forward_min_obs_dist:.2f}m > 1.2m) -> Trở lại chế độ tiến."
                )

        dynamic_min_speed = -0.8 if self.is_reversing else 0.0

        # Curvature-based speed profiling
        target_speed = self.target_speed
        if self.waypoints.shape[0] > 0:
            dx_all = self.waypoints[:, 0] - x0
            dy_all = self.waypoints[:, 1] - y0
            nearest_idx = int(np.argmin(dx_all**2 + dy_all**2))
            
            n_wps = len(self.waypoints)
            lookahead_indices = (np.arange(self.lookahead_wps) + nearest_idx) % n_wps
            future_curvatures = self.waypoint_curvatures[lookahead_indices]
            max_curve = float(np.max(future_curvatures))
            
            # Tính speed_factor dựa trên curvature (dùng lũy thừa 2 để tạo vùng chết tự nhiên, bỏ qua cua nhẹ)
            speed_factor = np.clip(1.0 - (max_curve / self.curve_threshold) ** 2, 0.0, 1.0)
            target_speed = self.min_speed_curve + (self.target_speed - self.min_speed_curve) * speed_factor

        # Điều tiết tốc độ dựa trên khoảng cách vật cản PHÍA TRƯỚC (chỉ giảm tốc khi quỹ đạo trước đó phát hiện chướng ngại vật thực sự)
        obs_speed_factor = 1.0
        if self.last_best_obs_cost >= 1.0:
            safe_braking_dist = max(1.5, v_cur * 0.5 + 0.5)
            min_safe_dist = 0.6
            if self.forward_min_obs_dist < safe_braking_dist:
                span = max(0.2, safe_braking_dist - min_safe_dist)
                obs_speed_factor = np.clip((self.forward_min_obs_dist - min_safe_dist) / span, 0.0, 1.0)
        
        # Tốc độ an toàn tối thiểu khi tránh chướng ngại vật gắt là 0.8 m/s
        min_speed_obs = 0.8
        target_speed = min_speed_obs + (target_speed - min_speed_obs) * obs_speed_factor

        # Nếu mất dữ liệu LiDAR quá 0.5s, giảm target speed để an toàn
        current_target_speed = 0.5 if lidar_lost else target_speed

        # Nếu đang ở chế độ lùi, đặt tốc độ mục tiêu âm để dẫn dắt MPPI lùi mượt mà
        if self.is_reversing:
            current_target_speed = -0.6

        # ── Giới hạn tốc độ tối đa động dựa trên khoảng cách tới vật cản phía trước và độ an toàn của quỹ đạo trước đó ──
        dynamic_max_speed = self.max_speed
        if self.forward_min_obs_dist < 3.0 and self.last_best_obs_cost >= 1.0:
            # Tuyến tính giảm tốc độ tối đa từ max_speed xuống 1.2 m/s khi khoảng cách từ 3.0m xuống 0.8m
            scale = np.clip((self.forward_min_obs_dist - 0.8) / 2.2, 0.0, 1.0)
            dynamic_max_speed = 1.2 + (self.max_speed - 1.2) * scale

        # ── Unstuck safety guard ─────────────────────────────────
        # Nếu xe đang dừng/chạy rất chậm nhưng đường thoáng phía trước, mà nominal control bị kẹt ở mức thấp
        # → reset nominal speed về current_target_speed để kích hoạt lại xe nhanh chóng
        if v_cur < 0.15 and self.forward_min_obs_dist > 1.5 and self.nominal_control[0, 0] < 0.5:
            self.get_logger().warn(
                f"[GUARD] Xe dang dung nhung duong thoang (forward_obs={self.forward_min_obs_dist:.2f}m) "
                f"→ reset nominal speed ve {current_target_speed:.2f} m/s",
                throttle_duration_sec=1.0
            )
            self.nominal_control[:, 0] = current_target_speed

        # ── Startup safety guard ─────────────────────────────────
        # Nếu xe đang chạy nhanh hơn nominal (sim khởi động với v lớn)
        # reset nominal về v_cur thực tế để tránh MPPI command sai.
        if (v_cur - self.nominal_control[0, 0]) > self.max_speed * 0.8:
            self.get_logger().warn(
                f"[GUARD] v_cur={v_cur:.2f} lệch xa nominal={self.nominal_control[0,0]:.2f} "
                "→ reset nominal về v_cur",
                throttle_duration_sec=2.0,
            )
            v_init = float(np.clip(v_cur, dynamic_min_speed, self.max_speed))
            self.nominal_control[:, 0] = v_init

        # ── Diagnostic: in 1 lần duy nhất để kiểm tra heading frame ─
        if not self._diag_done and self.waypoints.shape[0] > 0:
            dx_all = self.waypoints[:, 0] - x0
            dy_all = self.waypoints[:, 1] - y0
            nn_idx = int(np.argmin(dx_all**2 + dy_all**2))
            path_hdg = self.waypoint_headings[nn_idx]
            err_deg  = np.degrees((theta0 - path_hdg + np.pi) % (2*np.pi) - np.pi)
            self.get_logger().info(
                f"[DIAG] car_theta={np.degrees(theta0):.1f}° | "
                f"path_hdg={np.degrees(path_hdg):.1f}° | "
                f"heading_err={err_deg:.1f}° | "
                f"nearest_wp=({self.waypoints[nn_idx,0]:.2f},{self.waypoints[nn_idx,1]:.2f})"
            )
            self._diag_done = True

        # 2. Sinh nhiễu Gauss từ Generator riêng (OPT-3: thread-safe và nhanh hơn)
        noise = self.rng.normal(0.0, self.noise_sigma,
                                 (self.num_samples, self.horizon, 2))

        # 3. Chuỗi điều khiển nhiễu với clip biên cơ giới
        perturbed = self.nominal_control[None, :, :] + noise
        perturbed[:, :, 0] = np.clip(perturbed[:, :, 0], dynamic_min_speed, dynamic_max_speed)
        perturbed[:, :, 1] = np.clip(perturbed[:, :, 1], -self.max_steer, self.max_steer)

        # effective_noise: nhiễu thực sự được áp dụng sau khi clip (BUG-4)
        effective_noise = perturbed - self.nominal_control[None, :, :]

        # 4. Rollout: tích phân T bước song song trên N mẫu
        rollouts = np.zeros((self.num_samples, self.horizon + 1, 3))
        rollouts[:, 0, :] = state
        for t in range(self.horizon):
            rollouts[:, t + 1, :] = self._step(rollouts[:, t, :], perturbed[:, t, :])

        # 5. Tính chi phí
        costs = self._compute_cost(rollouts, perturbed, (x0, y0), theta0, current_target_speed)

        # 6. Cập nhật phân phối MPPI
        beta    = float(np.min(costs))
        
        # Adaptive Lambda (BUG-C)
        cost_std = float(np.std(costs))
        effective_lambda = max(self.lambda_, cost_std / 5.0)
        
        weights = np.exp(-(costs - beta) / effective_lambda)
        w_sum   = float(np.sum(weights))
        if w_sum < 1e-8:
            weights = np.ones(self.num_samples) / self.num_samples
        else:
            weights /= w_sum

        # U ← U + Σ_i( w_i · ε_i )
        self.nominal_control += np.sum(weights[:, None, None] * effective_noise, axis=0)

        # Clip nominal sau update
        self.nominal_control[:, 0] = np.clip(self.nominal_control[:, 0], dynamic_min_speed, dynamic_max_speed)
        self.nominal_control[:, 1] = np.clip(self.nominal_control[:, 1], -self.max_steer, self.max_steer)

        # 7. Lấy lệnh bước đầu tiên u_0
        opt_speed = float(self.nominal_control[0, 0])
        opt_steer = float(self.nominal_control[0, 1])

        # ── Safe Stop Guard: Phanh khẩn cấp nếu cận kề vật cản mà không tìm thấy đường đi an toàn ──
        best_idx = int(np.argmin(costs))
        best_obs_cost = self._current_obs_cost[best_idx]
        self.last_best_obs_cost = best_obs_cost
        
        # Ngưỡng khoảng cách dừng an toàn động dựa trên vận tốc hiện tại
        stop_threshold = max(0.9, v_cur * 0.4 + 0.3)
        if self.forward_min_obs_dist < stop_threshold and best_obs_cost >= 5.0:
            self.get_logger().warn(
                f"[SAFETY STOP] Cận kề vật cản và không tìm thấy đường đi an toàn! "
                f"(forward_obs={self.forward_min_obs_dist:.2f}m, threshold={stop_threshold:.2f}m, best_obs_cost={best_obs_cost:.1f}) -> Phanh dừng xe và kích hoạt đi lùi.",
                throttle_duration_sec=0.2
            )
            opt_speed = 0.0
            self.nominal_control[:, 0] = 0.0
            self.is_reversing = True

        # ── Receding horizon shift (warm-start) ─────────────────────
        # Dịch nominal_control 1 bước về trước
        self.nominal_control = np.roll(self.nominal_control, -1, axis=0)
        self.nominal_control[-1, 0] = current_target_speed
        self.nominal_control[-1, 1] = self.nominal_control[-2, 1]  # Giữ góc lái áp chót

        # 8. Log debug & hiệu năng chi tiết
        execution_time = (self.get_clock().now() - start_time).nanoseconds / 1e6
        
        n_eff = 1.0 / (np.sum(weights**2) + 1e-12)
        cost_std = float(np.std(costs))

        best_idx = int(np.argmin(costs))
        self._dbg_best_track = float(self.w_track * self._current_track_cost[best_idx])
        self._dbg_best_progress = float(self.w_progress * self._current_progress_cost[best_idx])
        self._dbg_best_speed = float(self.w_speed * self._current_speed_cost[best_idx])
        self._dbg_best_heading = float(self.w_heading * self._current_heading_cost[best_idx])
        self._dbg_best_obs = float(self.w_obstacle * self._current_obs_cost[best_idx])
        self._dbg_best_smooth = float(self.w_control * self._current_smooth_cost[best_idx])

        # Waypoint gần nhất hiện tại để chẩn đoán hướng
        dx_all = self.waypoints[:, 0] - x0
        dy_all = self.waypoints[:, 1] - y0
        nearest_idx = int(np.argmin(dx_all**2 + dy_all**2)) if self.waypoints.shape[0] > 0 else 0
        dist_to_wp = float(np.sqrt(dx_all[nearest_idx]**2 + dy_all[nearest_idx]**2)) if self.waypoints.shape[0] > 0 else 0.0
        path_hdg = self.waypoint_headings[nearest_idx] if self.waypoints.shape[0] > 0 else 0.0
        current_heading_err = (theta0 - path_hdg + np.pi) % (2*np.pi) - np.pi

        is_unstable = (n_eff < 5) or (beta > 5000.0) or (min_obs_dist < self.robot_radius + 0.05)
        
        should_log = (self.log_counter % 10 == 0)
        if is_unstable:
            if not hasattr(self, '_last_warn_counter'):
                self._last_warn_counter = 0
            if self.log_counter - self._last_warn_counter >= 4:
                should_log = True
                self._last_warn_counter = self.log_counter

        if should_log:
            cost_ratio = beta / (np.mean(costs) + 1e-6)
            
            log_msg = (
                f"[MPPI] {execution_time:.0f}ms | v={v_cur:.2f} → cmd={opt_speed:.2f} steer={opt_steer:.3f} | "
                f"n_eff={n_eff:.1f}/{self.num_samples} | cost_std={cost_std:.1f} | eff_lam={effective_lambda:.1f}\n"
                f"  cost min={beta:.1f} mean={np.mean(costs):.1f} ratio={cost_ratio:.3f} | "
                f"min_obs={min_obs_dist:.2f}m | wp=#{nearest_idx} (dist={dist_to_wp:.2f}m, err={np.degrees(current_heading_err):.1f}°)\n"
                f"  MEAN: track={self._dbg_track:.1f} prog={self.w_progress * np.mean(self._current_progress_cost):.1f} "
                f"speed={self._dbg_speed:.1f} hdg={self._dbg_heading:.1f} obs={self._dbg_obs:.1f} smooth={np.mean(self._current_smooth_cost) * self.w_control:.1f}\n"
                f"  BEST: track={self._dbg_best_track:.1f} prog={self._dbg_best_progress:.1f} "
                f"speed={self._dbg_best_speed:.1f} hdg={self._dbg_best_heading:.1f} obs={self._dbg_best_obs:.1f} smooth={self._dbg_best_smooth:.1f}"
            )
            
            if is_unstable:
                reasons = []
                if n_eff < 5:
                    reasons.append(f"low n_eff ({n_eff:.1f})")
                if beta > 5000.0:
                    reasons.append(f"collision cost ({beta:.1f})")
                if min_obs_dist < self.robot_radius + 0.05:
                    reasons.append(f"near wall ({min_obs_dist:.2f}m)")
                self.get_logger().warn(f"[UNSTABLE: {', '.join(reasons)}]\n{log_msg}")
            else:
                self.get_logger().info(log_msg)

        # 9. Gửi lệnh xuống actuator
        self._publish_drive(opt_speed, opt_steer)

        # 10. Hiển thị quỹ đạo danh nghĩa (BUG-3: dùng nominal rollout, không phải argmax mẫu)
        self._visualize_nominal_trajectory(state)

    def _publish_drive(self, speed: float, steer: float) -> None:
        drive = AckermannDriveStamped()
        drive.header.stamp         = self.get_clock().now().to_msg()
        drive.drive.speed          = float(speed)
        drive.drive.steering_angle = float(steer)
        self.pub_drive.publish(drive)

    # ─────────────────────────────────────────────────────────────────
    # Trực quan hóa
    # ─────────────────────────────────────────────────────────────────

    def _rollout_nominal(self, state: np.ndarray) -> np.ndarray:
        """Tích phân nominal_control từ state hiện tại. Trả về (T+1, 3)."""
        traj = np.zeros((self.horizon + 1, 3))
        traj[0] = state
        for t in range(self.horizon):
            v, d   = self.nominal_control[t]
            x, y, th = traj[t]
            traj[t + 1] = [
                x  + v * np.cos(th) * self.dt,
                y  + v * np.sin(th) * self.dt,
                th + (v * np.tan(d) / self.L) * self.dt,
            ]
        return traj

    def _visualize_nominal_trajectory(self, state: np.ndarray) -> None:
        traj = self._rollout_nominal(state)

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.ns              = "mppi_nominal_path"
        marker.id              = 0
        marker.type            = Marker.LINE_STRIP
        marker.action          = Marker.ADD
        marker.scale.x         = 0.06
        marker.color.r         = 1.0
        marker.color.a         = 1.0

        for pt in traj:
            p   = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            marker.points.append(p)

        self.pub_best_traj.publish(marker)

    def _publish_waypoints(self) -> None:
        if self.waypoints.shape[0] == 0:
            return
        now = self.get_clock().now().to_msg()
        ma  = MarkerArray()
        for i, pt in enumerate(self.waypoints):
            m = Marker()
            m.header.frame_id  = self.map_frame
            m.header.stamp     = now          # timestamp cho RViz2 ổn định
            m.id               = i
            m.type             = Marker.SPHERE
            m.action           = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.1
            m.color.a = 1.0
            m.color.r = m.color.g = m.color.b = 0.6
            m.pose.position.x  = float(pt[0])
            m.pose.position.y  = float(pt[1])
            ma.markers.append(m)
        self.pub_waypoints.publish(ma)

    # ─────────────────────────────────────────────────────────────────
    # Virtual obstacle callbacks
    # ─────────────────────────────────────────────────────────────────

    def clicked_point_callback(self, msg: PointStamped) -> None:
        x_click = msg.point.x
        y_click = msg.point.y
        frame_click = msg.header.frame_id

        if frame_click != self.map_frame:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    frame_click,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05)
                )
                tx = tf.transform.translation.x
                ty = tf.transform.translation.y
                q = tf.transform.rotation
                yaw = np.arctan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                )
                cos_y = np.cos(yaw)
                sin_y = np.sin(yaw)
                x_map = cos_y * x_click - sin_y * y_click + tx
                y_map = sin_y * x_click + cos_y * y_click + ty
            except Exception as e:
                self.get_logger().error(f"Failed to transform clicked point: {e}")
                return
        else:
            x_map = x_click
            y_map = y_click

        # Toggle/xóa vật cản nếu click gần vật cản ảo cũ (< 0.6m)
        clicked_near = False
        if len(self.virtual_obstacles) > 0:
            dists = np.linalg.norm(np.array(self.virtual_obstacles) - np.array([x_map, y_map]), axis=1)
            min_idx = np.argmin(dists)
            if dists[min_idx] < 0.6:
                self.virtual_obstacles.pop(min_idx)
                self.get_logger().info(f"Removed virtual obstacle at ({x_map:.2f}, {y_map:.2f})")
                clicked_near = True

        if not clicked_near:
            self.virtual_obstacles.append([x_map, y_map])
            self.get_logger().info(f"Added virtual obstacle at ({x_map:.2f}, {y_map:.2f})")

        self._publish_virtual_obstacles()

    def clear_virtual_obstacles_callback(self, msg) -> None:
        self.virtual_obstacles = []
        self.get_logger().info("Cleared all virtual obstacles.")
        self._publish_virtual_obstacles()

    def _publish_virtual_obstacles(self) -> None:
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "virtual_obstacles"
        marker.id = 1
        marker.type = Marker.SPHERE_LIST
        
        if len(self.virtual_obstacles) == 0:
            marker.action = Marker.DELETE
        else:
            marker.action = Marker.ADD
            marker.scale.x = self.robot_radius * 2.0
            marker.scale.y = self.robot_radius * 2.0
            marker.scale.z = self.robot_radius * 2.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            for obs in self.virtual_obstacles:
                p = Point()
                p.x = float(obs[0])
                p.y = float(obs[1])
                p.z = 0.0
                marker.points.append(p)
                
        self.pub_virtual_obs.publish(marker)


# ─────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = MPPIController()
    
    # Sử dụng MultiThreadedExecutor để chạy các callback sensor (odom, lidar) song song
    # độc lập với luồng chạy control_loop của MPPI.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_drive(0.0, 0.0)   # phanh khẩn cấp dừng xe khi tắt node
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()