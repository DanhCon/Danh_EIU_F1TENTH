#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import qos_profile_sensor_data


class BasicDisparityExtender(Node):
    def __init__(self):
        super().__init__('basic_disparity_node')

        # --- THAM SỐ CƠ BẢN ---
        self.fov_min             = math.radians(-120.0)
        self.fov_max             = math.radians(120.0)
        self.car_width           = 0.6
        self.disparity_threshold = 0.06
        self.safe_dist           = 0.4
        self.prev_angle          = 0.0
        self.SMOOTH_ALPHA        = 0.35

        # --- THAM SỐ MỚI: AN TOÀN ---
        # Emergency brake: nếu vùng ±FRONT_CONE phía trước có ray < BRAKE_DIST → dừng hẳn
        self.FRONT_CONE_DEG  = 30.0          # ±30° tính từ thẳng trước
        self.BRAKE_DIST      = 0.5           # (m) dừng khẩn cấp
        self.CREEP_DIST      = 0.8           # (m) chạy chậm
        self.CREEP_SPEED     = 0.5           # (m/s) tốc độ bò khi gần tường phía trước

        # Gap tại boundary FOV (-120° hoặc +120°) thường là khoảng trống hông/sau xe
        # → loại bỏ nếu angle trung tâm của gap lệch quá xa khỏi 0°
        self.MAX_GAP_CENTER_ANGLE = math.radians(75.0)  # loại gap nếu center > 75° khỏi trục thẳng

        # Deadzone góc lái nhỏ → bỏ để tránh xe đi thẳng khi cần rẽ
        self.ANGLE_DEADZONE  = math.radians(1.5)

        # --- VIRTUAL OBSTACLES ---
        self.virtual_obstacles = []
        self.car_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}

        # --- ROS2 ---
        self.scan_sub  = self.create_subscription(LaserScan,    '/scan',              self.lidar_callback,  qos_profile_sensor_data)
        self.odom_sub  = self.create_subscription(Odometry,     '/ego_racecar/odom',  self.odom_callback,   10)
        self.click_sub = self.create_subscription(PointStamped, '/clicked_point',     self.click_callback,  10)
        self.clear_sub = self.create_subscription(PoseStamped,  '/goal_pose',         self.clear_callback,  10)

        self.drive_pub  = self.create_publisher(AckermannDriveStamped, '/drive',                    10)
        self.marker_pub = self.create_publisher(MarkerArray,           '/virtual_obstacles_markers', 10)

        self.create_timer(0.1, self.publish_markers)
        self.get_logger().info("🏎️  Disparity Extender [FIXED] Ready!")

    # ═══════════════════════════════════════════════════════
    #  ODOMETRY & VIRTUAL OBSTACLES (giữ nguyên)
    # ═══════════════════════════════════════════════════════
    def odom_callback(self, msg: Odometry):
        self.car_pose['x'] = msg.pose.pose.position.x
        self.car_pose['y'] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.car_pose['yaw'] = math.atan2(siny_cosp, cosy_cosp)

    def click_callback(self, msg: PointStamped):
        new_obs = {'x': msg.point.x, 'y': msg.point.y, 'r': 0.2}
        self.virtual_obstacles.append(new_obs)
        self.get_logger().info(f"📍 Thêm vật cản: ({new_obs['x']:.2f}, {new_obs['y']:.2f})")

    def clear_callback(self, msg: PoseStamped):
        self.virtual_obstacles.clear()
        ma = MarkerArray()
        dm = Marker(); dm.action = Marker.DELETEALL
        ma.markers.append(dm)
        self.marker_pub.publish(ma)
        self.get_logger().warn("🗑️  Đã xóa toàn bộ vật cản ảo")

    def inject_virtual_obstacles(self, ranges, angle_min, angle_increment):
        if not self.virtual_obstacles:
            return ranges
        new_ranges = np.array(ranges)
        angles = angle_min + np.arange(len(new_ranges)) * angle_increment
        for obs in self.virtual_obstacles:
            dx      = obs['x'] - self.car_pose['x']
            dy      = obs['y'] - self.car_pose['y']
            x_local = dx * math.cos(-self.car_pose['yaw']) - dy * math.sin(-self.car_pose['yaw'])
            y_local = dx * math.sin(-self.car_pose['yaw']) + dy * math.cos(-self.car_pose['yaw'])
            dist    = math.hypot(x_local, y_local)
            if x_local < 0 or dist > 10.0:
                continue
            theta_c   = math.atan2(y_local, x_local)
            delta_t   = math.asin(min(1.0, obs['r'] / dist))
            mask      = (angles > theta_c - delta_t) & (angles < theta_c + delta_t)
            virt_dist = dist - obs['r']
            for i in np.where(mask)[0]:
                if virt_dist < new_ranges[i]:
                    new_ranges[i] = max(0.0, virt_dist)
        return new_ranges.tolist()

    # ═══════════════════════════════════════════════════════
    #  EMERGENCY BRAKE CHECK  ← FIX #1
    # ═══════════════════════════════════════════════════════
    def check_front_clearance(self, ranges, angle_min, angle_increment):
        """
        Trả về khoảng cách nhỏ nhất trong vùng ±FRONT_CONE_DEG phía trước.
        Dùng để quyết định có cần brake khẩn cấp không.
        """
        cone_rad  = math.radians(self.FRONT_CONE_DEG)
        n         = len(ranges)
        min_dist  = float('inf')
        for i, r in enumerate(ranges):
            angle = angle_min + i * angle_increment
            if abs(angle) <= cone_rad and r > 0.01:
                if r < min_dist:
                    min_dist = r
        return min_dist if min_dist != float('inf') else 5.0

    # ═══════════════════════════════════════════════════════
    #  CORE PIPELINE
    # ═══════════════════════════════════════════════════════
    def preprocess_lidar(self, ranges, max_dist):
        window_size = 5
        proc = [min(r, max_dist) if not math.isinf(r) and not math.isnan(r) else max_dist for r in ranges]
        smooth = [0.0] * len(proc)
        for i in range(len(proc)):
            s = max(0, i - window_size // 2)
            e = min(len(proc), i + window_size // 2 + 1)
            smooth[i] = sum(proc[s:e]) / (e - s)
        return smooth

    def extend_disparities(self, ranges, angle_increment):
        original = ranges.copy()
        filtered = ranges.copy()
        CLOSE_DIST, FAR_MULT = 2.0, 4.0
        for i in range(len(original) - 1):
            disparity = abs(original[i] - original[i + 1])
            near_dist = min(original[i], original[i + 1])
            threshold = self.disparity_threshold if near_dist < CLOSE_DIST else self.disparity_threshold * FAR_MULT
            if disparity > threshold:
                min_val     = min(original[i], original[i + 1])
                bubble_ang  = 2 * math.atan(self.car_width / (2 * max(min_val, 0.1)))
                bubble_rays = int(bubble_ang / angle_increment)
                if original[i] < original[i + 1]:
                    for j in range(max(0, i - bubble_rays), i + 1):
                        filtered[j] = 0.0
                else:
                    for j in range(i + 1, min(len(filtered), i + 2 + bubble_rays)):
                        filtered[j] = 0.0
        return filtered

    def find_best_gap(self, ranges, angle_min, angle_increment):
        """
        FIX #2: Tìm gap tốt nhất có tính đến hướng của gap.
        
        Loại bỏ gap bắt đầu/kết thúc tại boundary FOV nếu center của gap
        lệch quá xa khỏi trục thẳng phía trước (0°).
        
        Score = gap_length * cos²(angle_center)
        → ưu tiên gap rộng VÀ hướng về phía trước.
        """
        n          = len(ranges)
        best_score = -1.0
        best_start = 0
        best_end   = 0

        curr_start = -1
        curr_len   = 0

        def evaluate_gap(gs, ge):
            nonlocal best_score, best_start, best_end
            length = ge - gs + 1
            if length < 3:
                return
            center_idx   = (gs + ge) // 2
            center_angle = angle_min + center_idx * angle_increment

            # FIX #2: Loại gap có center quá lệch so với trục thẳng
            if abs(center_angle) > self.MAX_GAP_CENTER_ANGLE:
                return

            # Score ưu tiên gap hướng phía trước
            score = length * (math.cos(center_angle) ** 2)
            if score > best_score:
                best_score = score
                best_start = gs
                best_end   = ge

        for i, r in enumerate(ranges):
            if r > self.safe_dist:
                if curr_start == -1:
                    curr_start = i
                curr_len += 1
            else:
                if curr_len > 0:
                    evaluate_gap(curr_start, curr_start + curr_len - 1)
                curr_start, curr_len = -1, 0
        if curr_len > 0:
            evaluate_gap(curr_start, curr_start + curr_len - 1)

        return best_start, best_end, best_score > 0

    def find_best_point(self, start_idx, end_idx, ranges):
        """
        FIX #3: Chọn điểm xa nhất trong vùng rộng nhất của gap,
        có bias nhẹ về phía center để tránh edge.
        """
        if start_idx >= end_idx:
            return (start_idx + end_idx) // 2

        sub   = ranges[start_idx: end_idx + 1]
        max_v = max(sub)
        thr   = max_v * 0.5      # ngưỡng cao hơn → chỉ lấy vùng thực sự rộng

        best_s = best_e = 0
        best_l = 0
        c_s    = -1
        c_l    = 0
        for i, v in enumerate(sub):
            if v >= thr:
                if c_s == -1:
                    c_s = i
                c_l += 1
            else:
                if c_l > best_l:
                    best_l, best_s, best_e = c_l, c_s, c_s + c_l - 1
                c_s, c_l = -1, 0
        if c_l > best_l:
            best_s, best_e = c_s, c_s + c_l - 1

        # Lấy center của sub-gap tốt nhất
        return start_idx + (best_s + best_e) // 2

    # ═══════════════════════════════════════════════════════
    #  MAIN CALLBACK
    # ═══════════════════════════════════════════════════════
    def lidar_callback(self, data: LaserScan):
        s_idx = int((self.fov_min - data.angle_min) / data.angle_increment)
        e_idx = min(int((self.fov_max - data.angle_min) / data.angle_increment), len(data.ranges) - 1)

        raw_ranges      = list(data.ranges[s_idx:e_idx])
        injected_ranges = self.inject_virtual_obstacles(raw_ranges, self.fov_min, data.angle_increment)
        ranges          = self.preprocess_lidar(injected_ranges, 5.0)

        # ── FIX #1: Kiểm tra khoảng trống phía trước TRƯỚC KHI extend ──
        front_clear = self.check_front_clearance(ranges, self.fov_min, data.angle_increment)

        if front_clear < self.BRAKE_DIST:
            # Tường sát phía trước → dừng khẩn cấp, giữ nguyên góc lái
            self.get_logger().warn(
                f"🛑 EMERGENCY BRAKE: front_clear={front_clear:.2f}m < {self.BRAKE_DIST}m"
            )
            self.publish_drive(self.prev_angle, 0.0)
            return

        ranges = self.extend_disparities(ranges, data.angle_increment)

        # ── FIX #2: Tìm gap có hướng hợp lệ ──
        gap_s, gap_e, found = self.find_best_gap(ranges, self.fov_min, data.angle_increment)

        if not found:
            # Không tìm được gap hợp lệ → dừng
            self.get_logger().warn("⚠️  Không có gap hợp lệ hướng về phía trước → DỪNG")
            self.publish_drive(0.0, 0.0)
            return

        # ── FIX #3: Best point ──
        best_idx   = self.find_best_point(gap_s, gap_e, ranges)
        raw_angle  = self.fov_min + best_idx * data.angle_increment

        if abs(raw_angle) < self.ANGLE_DEADZONE:
            raw_angle = 0.0

        smooth_angle    = self.SMOOTH_ALPHA * raw_angle + (1 - self.SMOOTH_ALPHA) * self.prev_angle
        self.prev_angle = smooth_angle

        # ── FIX #4: Tốc độ theo cả góc lái VÀ khoảng trống phía trước ──
        if front_clear < self.CREEP_DIST:
            speed = self.CREEP_SPEED
        elif abs(smooth_angle) < math.radians(10.0):
            speed = 4.0
        elif abs(smooth_angle) < math.radians(20.0):
            speed = 3.5
        else:
            speed = 2.0

        self.publish_drive(smooth_angle, speed)

    # ═══════════════════════════════════════════════════════
    #  PUBLISH
    # ═══════════════════════════════════════════════════════
    def publish_drive(self, angle, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp         = self.get_clock().now().to_msg()
        msg.header.frame_id      = "base_link"
        msg.drive.steering_angle = float(max(-0.43, min(0.43, angle)))
        msg.drive.speed          = float(speed)
        self.drive_pub.publish(msg)

    def publish_markers(self):
        if not self.virtual_obstacles:
            return
        ma = MarkerArray()
        for i, obs in enumerate(self.virtual_obstacles):
            m = Marker()
            m.header.frame_id = "map"
            m.id              = i
            m.type            = Marker.CYLINDER
            m.action          = Marker.ADD
            m.pose.position.x = obs['x']
            m.pose.position.y = obs['y']
            m.pose.position.z = 0.2
            m.scale.x         = obs['r'] * 2
            m.scale.y         = obs['r'] * 2
            m.scale.z         = 0.4
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 0.6
            ma.markers.append(m)
        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = BasicDisparityExtender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()