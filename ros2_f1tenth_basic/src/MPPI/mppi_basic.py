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
"""

import csv

import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
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
        self.noise_sigma = np.array([0.15, 0.15])

        # Temperature λ: càng lớn → phân phối trọng số càng mịn (ít tập trung vào mẫu tốt nhất)
        self.lambda_ = 0.45

        # ── Giới hạn cơ giới ─────────────────────────────────────────
        self.max_speed = 1.5
        self.min_speed = 0.0
        self.max_steer = 0.35   # ~24 độ

        # ── Trọng số hàm chi phí ─────────────────────────────────────
        self.w_track    = 20.0   # Bám đường raceline
        self.w_control  = 2.5    # Làm mịn lệnh điều khiển
        self.w_obstacle = 65.0   # Tránh vật cản

        # Bán kính an toàn của xe (m)
        self.robot_radius = 0.35

        # Tốc độ warm-start cho bước cuối horizon
        self.target_speed = 3.5

        # Kích thước cửa sổ waypoint cục bộ dùng trong track cost
        # (thay vì tính với toàn bộ 10k+ điểm — tránh OOM)
        self.wp_window = 60

        # ── Chuỗi điều khiển danh nghĩa U: (T, 2) → [speed, steer] ──
        self.nominal_control = np.zeros((self.horizon, 2))
        self.nominal_control[:, 0] = self.target_speed

        # Vật cản trong MAP frame (cập nhật từ lidar_callback qua TF)
        self.map_obstacles = np.zeros((0, 2))

        # ── TF ───────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.car_frame   = "ego_racecar/base_link"
        self.map_frame   = "map"

        # ── ROS 2 pub/sub ─────────────────────────────────────────────
        self.sub_odom  = self.create_subscription(Odometry,  "ego_racecar/odom", self.odom_callback,  10)
        self.sub_laser = self.create_subscription(LaserScan, "/scan",             self.lidar_callback, 10)

        self.pub_drive     = self.create_publisher(AckermannDriveStamped, "/drive",                 10)
        self.pub_best_traj = self.create_publisher(Marker,                "/mppi_best_trajectory",  10)
        self.pub_waypoints = self.create_publisher(MarkerArray,           "/publish_full_waypoint", 10)

        # ── Nạp waypoints ─────────────────────────────────────────────
        self.waypoints = np.zeros((0, 2))
        csv_path = (
            "/home/danh/ros2_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        )
        self._load_waypoints(csv_path)
        self._publish_waypoints()

        self.log_counter = 0
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
        except Exception as e:
            self.get_logger().error(f"Cannot load CSV: {e}")

    # ─────────────────────────────────────────────────────────────────
    # LiDAR callback — chuyển obstacles sang MAP frame ngay lúc nhận
    # ─────────────────────────────────────────────────────────────────

    def lidar_callback(self, msg: LaserScan) -> None:
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # Lọc nhiễu: loại bỏ điểm quét quá gần, quá xa, NaN, Inf
        valid = np.isfinite(ranges) & (ranges > 0.15) & (ranges < 5.0)
        r   = ranges[valid]
        phi = angles[valid]

        # Tọa độ trong base_link
        x_car = r * np.cos(phi)
        y_car = r * np.sin(phi)

        # Chuyển sang MAP frame qua TF (FIX BUG-1: nhất quán khung tọa độ)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.car_frame,
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
        return np.stack(
            [
                x  + v * np.cos(th) * dt,
                y  + v * np.sin(th) * dt,
                th + (v * np.tan(d) / self.L) * dt,
            ],
            axis=1,
        )

    # ─────────────────────────────────────────────────────────────────
    # Lấy cửa sổ waypoint gần nhất (OPT-1)
    # ─────────────────────────────────────────────────────────────────

    def _local_waypoints(self, x: float, y: float) -> np.ndarray:
        """Trả về wp_window điểm waypoint gần xe nhất (vòng tròn)."""
        if self.waypoints.shape[0] == 0:
            return self.waypoints
        dx      = self.waypoints[:, 0] - x
        dy      = self.waypoints[:, 1] - y
        nearest = int(np.argmin(dx * dx + dy * dy))
        n       = len(self.waypoints)
        idx     = np.arange(nearest, nearest + self.wp_window) % n
        return self.waypoints[idx]

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
    ) -> np.ndarray:
        # Tọa độ tương lai (N, T, 2) — bỏ bước t=0 (trạng thái hiện tại)
        pts = state_rollouts[:, 1:, :2]

        # ── 1. Track cost: khoảng cách bình phương đến waypoint gần nhất ──
        local_wps = self._local_waypoints(pos[0], pos[1])
        if local_wps.shape[0] == 0:
            return np.full(self.num_samples, 1e5)

        # (N, T, W, 2) → norm → (N, T, W) → min → (N, T) → sum → (N,)
        delta_wp   = pts[:, :, None, :] - local_wps[None, None, :, :]
        dist_wp    = np.linalg.norm(delta_wp, axis=3)
        track_cost = np.sum(np.min(dist_wp, axis=2) ** 2, axis=1)

        # ── 2. Smoothness cost: phạt thay đổi lệnh điều khiển đột ngột ──
        ctrl_diff    = perturbed_controls[:, 1:, :] - perturbed_controls[:, :-1, :]
        smooth_cost  = np.sum(ctrl_diff ** 2, axis=(1, 2))  # gộp speed+steer

        # ── 3. Obstacle cost: phạt tiến gần vật cản (BUG-1 đã sửa) ──
        # pts và map_obstacles đều ở MAP frame → so sánh trực tiếp, không cần xoay thêm
        obs_cost = np.zeros(self.num_samples)
        if self.map_obstacles.shape[0] > 0:
            # (N, T, M, 2) → norm → (N, T, M)
            delta_obs = pts[:, :, None, :] - self.map_obstacles[None, None, :, :]
            dists     = np.linalg.norm(delta_obs, axis=3)
            in_danger = dists < self.robot_radius
            # Chi phí nghịch đảo: càng gần vật cản → chi phí càng lớn
            per_step  = np.sum(1.0 / (dists + 1e-3) * in_danger, axis=2)
            obs_cost  = np.sum(per_step, axis=1)

        return (
            self.w_track    * track_cost  +
            self.w_control  * smooth_cost +
            self.w_obstacle * obs_cost
        )

    # ─────────────────────────────────────────────────────────────────
    # Vòng lặp điều khiển chính (20 Hz)
    # ─────────────────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry) -> None:
        self.log_counter += 1

        # 1. Đọc trạng thái hiện tại
        x0     = msg.pose.pose.position.x
        y0     = msg.pose.pose.position.y
        theta0 = self._quat_to_yaw(msg.pose.pose.orientation)
        v_cur  = msg.twist.twist.linear.x
        state  = np.array([x0, y0, theta0])

        # 2. Sinh nhiễu Gauss: (N, T, 2)
        noise = np.random.normal(0.0, self.noise_sigma,
                                 (self.num_samples, self.horizon, 2))

        # 3. Chuỗi điều khiển nhiễu với clip biên cơ giới
        perturbed = self.nominal_control[None, :, :] + noise
        perturbed[:, :, 0] = np.clip(perturbed[:, :, 0], self.min_speed, self.max_speed)
        perturbed[:, :, 1] = np.clip(perturbed[:, :, 1], -self.max_steer, self.max_steer)

        # effective_noise: nhiễu thực sự được áp dụng sau khi clip (BUG-4)
        # Đảm bảo rollout và weight update dùng cùng lượng perturbation
        effective_noise = perturbed - self.nominal_control[None, :, :]

        # 4. Rollout: tích phân T bước song song trên N mẫu
        rollouts = np.zeros((self.num_samples, self.horizon + 1, 3))
        rollouts[:, 0, :] = state
        for t in range(self.horizon):
            rollouts[:, t + 1, :] = self._step(rollouts[:, t, :], perturbed[:, t, :])

        # 5. Tính chi phí
        costs = self._compute_cost(rollouts, perturbed, (x0, y0))

        # 6. Cập nhật phân phối MPPI (Williams et al. 2018)
        # Trick ổn định số: trừ min_cost trước khi exp để tránh underflow
        beta    = float(np.min(costs))
        weights = np.exp(-(costs - beta) / self.lambda_)
        w_sum   = float(np.sum(weights))
        if w_sum < 1e-8:
            weights = np.ones(self.num_samples) / self.num_samples
        else:
            weights /= w_sum

        # U ← U + Σ_i( w_i · ε_i )  — công thức chuẩn từ paper
        self.nominal_control += np.sum(weights[:, None, None] * effective_noise, axis=0)

        # Clip nominal sau update
        self.nominal_control[:, 0] = np.clip(self.nominal_control[:, 0], self.min_speed, self.max_speed)
        self.nominal_control[:, 1] = np.clip(self.nominal_control[:, 1], -self.max_steer, self.max_steer)

        # 7. Lấy lệnh bước đầu tiên u_0
        opt_speed = float(self.nominal_control[0, 0])
        opt_steer = float(self.nominal_control[0, 1])

        # 8. Log debug (10 chu kỳ/lần)
        if self.log_counter % 10 == 0:
            self.get_logger().info(
                f"[MPPI] v={v_cur:.2f} | cmd: v={opt_speed:.2f}, steer={opt_steer:.3f} | "
                f"cost: min={beta:.1f}, mean={np.mean(costs):.1f} | "
                f"obs={self.map_obstacles.shape[0]}"
            )

        # 9. Gửi lệnh xuống actuator
        drive = AckermannDriveStamped()
        drive.header.stamp         = self.get_clock().now().to_msg()
        drive.drive.speed          = opt_speed
        drive.drive.steering_angle = opt_steer
        self.pub_drive.publish(drive)

        # 10. Hiển thị quỹ đạo danh nghĩa (BUG-3: dùng nominal rollout, không phải argmax mẫu)
        self._visualize_nominal_trajectory(state)

        # 11. Dịch chuyển Receding Horizon (BUG-2: lưu last_steer TRƯỚC khi shift)
        last_steer = float(self.nominal_control[-1, 1])   # lưu trước
        self.nominal_control[:-1] = self.nominal_control[1:]
        self.nominal_control[-1, 0] = self.target_speed
        self.nominal_control[-1, 1] = last_steer          # phục hồi sau

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


# ─────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = MPPIController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()