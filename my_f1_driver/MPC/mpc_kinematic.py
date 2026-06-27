#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import csv
import numpy as np
from copy import deepcopy

from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import Point

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

def normalize_angle(angle):
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle

class SimKinematicMPCNode(Node):
    def __init__(self):
        super().__init__("sim_kinematic_mpc_node")

        # ========================================================
        # 1. THAM SỐ KINEMATIC (DÙNG CHUNG CHO SIM VÀ THỰC TẾ)
        # ========================================================
        self.lf = 0.158     
        self.lr = 0.171     
        self.L = self.lf + self.lr
        self.Ts = 0.05      # Tần số tính toán 20Hz
        self.hz = 15        # Tầm nhìn trước (Horizons)
        
        self.Q = np.array([[10.0, 0.0], 
                           [0.0, 100.0]])  
        self.S = np.array([[10.0, 0.0], 
                           [0.0, 100.0]])  
        self.R = np.array([[300.0]])      

        # ========================================================
        # 2. KHỞI TẠO ROS 2 CHO SIMULATOR
        # ========================================================
        self.U1 = 0.0 
        self.start_index = None
        self.waypoints = [] 

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # [THAY ĐỔI 1] Trả lại namespace ego_racecar cho simulator
        self.car_frame = "ego_racecar/base_link" 
        self.map_frame = "map"

        # [THAY ĐỔI 2] Odom trong sim nằm ở namespace ego_racecar
        self.sub_odom = self.create_subscription(Odometry, "ego_racecar/odom", self.odom_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.pub_marker_path = self.create_publisher(MarkerArray, "/publish_full_waypoint", 10)
        self.pub_mpc_ref = self.create_publisher(Marker, "/mpc_lookahead_points", 10)

        # [THAY ĐỔI 3] Đường dẫn tuyệt đối đến file CSV trong container/workspace sim
        csv_path = "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        self.load_waypoints(csv_path)
        self.publish_full_waypoint()

        self.last_mpc_time = self.get_clock().now()
        self.get_logger().info("Kinematic MPC (Bản Simulation) đã khởi động!")

    # ========================================================
    # MÔ HÌNH TOÁN HỌC (KINEMATIC ZOH)
    # ========================================================
    def calculate_kinematic_state_space(self, v_x):
        # Tránh chia 0 khi xe đang dừng (trong sim có thể để nhỏ hơn xe thật)
        v_x = max(v_x, 0.5) 
        
        Ad = np.array([
            [1.0, v_x * self.Ts],
            [0.0, 1.0]
        ])
        Bd = np.array([
            [(v_x**2 * self.Ts**2) / (2 * self.L)],
            [(v_x * self.Ts) / self.L]
        ])
        Cd = np.eye(2)
        return Ad, Bd, Cd

    def mpc_simplification(self, Ad, Bd, Cd):
        A_aug = np.block([[Ad, Bd], [np.zeros((1, 2)), np.eye(1)]])
        B_aug = np.block([[Bd], [np.eye(1)]])
        C_aug = np.block([[Cd, np.zeros((2, 1))]])

        CQC = C_aug.T @ self.Q @ C_aug
        CSC = C_aug.T @ self.S @ C_aug
        QC = self.Q @ C_aug
        SC = self.S @ C_aug

        Qdb = np.zeros((3 * self.hz, 3 * self.hz))
        Tdb = np.zeros((2 * self.hz, 3 * self.hz))
        Rdb = np.zeros((1 * self.hz, 1 * self.hz))
        Cdb = np.zeros((3 * self.hz, 1 * self.hz))
        Adc = np.zeros((3 * self.hz, 3))

        for i in range(self.hz):
            if i == self.hz - 1:
                Qdb[3*i:3*i+3, 3*i:3*i+3] = CSC
                Tdb[2*i:2*i+2, 3*i:3*i+3] = SC
            else:
                Qdb[3*i:3*i+3, 3*i:3*i+3] = CQC
                Tdb[2*i:2*i+2, 3*i:3*i+3] = QC
            Rdb[i, i] = self.R[0, 0]
            for j in range(self.hz):
                if j <= i:
                    Cdb[3*i:3*i+3, j:j+1] = np.linalg.matrix_power(A_aug, i-j) @ B_aug
            Adc[3*i:3*i+3, :] = np.linalg.matrix_power(A_aug, i+1)

        Hdb = Cdb.T @ Qdb @ Cdb + Rdb
        Fdbt = np.vstack((Adc.T @ Qdb @ Cdb, -Tdb @ Cdb))
        return Hdb, Fdbt, Cdb, Adc

    def odom_callback(self, msg: Odometry):
        if not self.waypoints: return

        current_time = self.get_clock().now()
        dt = (current_time - self.last_mpc_time).nanoseconds / 1e9
        if dt < self.Ts: return  
        self.last_mpc_time = current_time

        # Ở mô phỏng, v_x rất mượt và chính xác
        v_x = msg.twist.twist.linear.x

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.car_frame, rclpy.time.Time())
            rx = transform.transform.translation.x
            ry = transform.transform.translation.y
            q = transform.transform.rotation
            r_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        except TransformException: return

        min_dist = float("inf")
        nearest_idx = self.start_index if self.start_index is not None else 0
        
        search_range = range(len(self.waypoints)) if self.start_index is None else range(nearest_idx, nearest_idx + 20)
        for i_raw in search_range:
            i = i_raw % len(self.waypoints)
            d = math.hypot(rx - self.waypoints[i][0], ry - self.waypoints[i][1])
            if d < min_dist:
                min_dist = d
                self.start_index = i

        nearest_idx = self.start_index
        wp_x, wp_y, wp_yaw = self.waypoints[nearest_idx]

        dx = rx - wp_x
        dy = ry - wp_y
        e_y = -math.sin(wp_yaw) * dx + math.cos(wp_yaw) * dy
        e_psi = normalize_angle(r_yaw - wp_yaw)

        Ad, Bd, Cd = self.calculate_kinematic_state_space(v_x)
        Hdb, Fdbt, Cdb, Adc = self.mpc_simplification(Ad, Bd, Cd)

        states = np.array([e_y, e_psi])
        x_aug_t = np.concatenate((states, [self.U1]))
        
        r_list = []
        ref_global_points = []
        step_dist = max(v_x, 0.5) * self.Ts 
        curr_idx = nearest_idx
        dist_accum = 0.0
        
        for i in range(1, self.hz + 1):
            target_dist = i * step_dist 
            while True:
                next_idx = (curr_idx + 1) % len(self.waypoints)
                p1 = self.waypoints[curr_idx]
                p2 = self.waypoints[next_idx]
                segment_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                
                if dist_accum + segment_len >= target_dist:
                    ratio = (target_dist - dist_accum) / segment_len if segment_len > 0 else 0.0
                    fx = p1[0] + ratio * (p2[0] - p1[0])
                    fy = p1[1] + ratio * (p2[1] - p1[1])
                    fyaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    break
                else:
                    dist_accum += segment_len
                    curr_idx = next_idx
                    
            ref_global_points.append((fx, fy))
            
            fdx = fx - wp_x
            fdy = fy - wp_y
            future_e_y = -math.sin(wp_yaw) * fdx + math.cos(wp_yaw) * fdy
            future_e_psi = normalize_angle(fyaw - wp_yaw)
            
            r_list.extend([future_e_y, future_e_psi])
            
        r_vector = np.array(r_list)

        self.publish_mpc_reference(ref_global_points)
        ft_input = np.concatenate((x_aug_t, r_vector))
        ft = Fdbt.T @ ft_input
        
        try:
            du = -np.linalg.inv(Hdb) @ ft
            self.U1 = self.U1 + du[0]
            max_steer = 0.4 
            self.U1 = np.clip(self.U1, -max_steer, max_steer) 
        except np.linalg.LinAlgError: return

        # ========================================================
        # [THAY ĐỔI 4] LOGIC TỐC ĐỘ BỐC HƠN CHO SIMULATOR
        # ========================================================
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(self.U1)

        # Trong sim, xe có độ bám dính tuyệt đối, cho phép nâng base_speed lên 6.0 m/s
        base_speed = 6.0  
        min_speed = 6.0   
        
        target_speed = base_speed - (abs(self.U1) / max_steer) * (base_speed - min_speed)
        drive_msg.drive.speed = max(min_speed, min(base_speed, target_speed))
            
        self.pub_drive.publish(drive_msg)
        
    def load_waypoints(self, filename):
        raw_waypoints = []
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row: continue
                    line_data = row[0].split() if len(row) == 1 else row
                    try:
                        raw_waypoints.append([float(line_data[0]), float(line_data[1])])
                    except ValueError: continue
            
            smoothed = self.smooth_path(raw_waypoints)
            self.waypoints = []
            for i in range(len(smoothed)):
                p1 = smoothed[i]
                p2 = smoothed[(i + 1) % len(smoothed)]
                yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                self.waypoints.append([p1[0], p1[1], yaw])
        except Exception as e:
            self.get_logger().error(f"LỖI ĐỌC FILE: {e}")

    def smooth_path(self, path, weight_data=0.5, weight_smooth=0.2, tolerance=0.00001):
        new_path = deepcopy(path)
        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path) - 1):
                aux_x, aux_y = new_path[i][0], new_path[i][1]
                new_path[i][0] += weight_data * (path[i][0] - new_path[i][0]) + \
                                  weight_smooth * (new_path[i-1][0] + new_path[i+1][0] - 2.0 * new_path[i][0])
                new_path[i][1] += weight_data * (path[i][1] - new_path[i][1]) + \
                                  weight_smooth * (new_path[i-1][1] + new_path[i+1][1] - 2.0 * new_path[i][1])
                change += abs(aux_x - new_path[i][0]) + abs(aux_y - new_path[i][1])
        return new_path

    def publish_mpc_reference(self, points_list):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "mpc_reference"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST 
        marker.action = Marker.ADD
        marker.scale.x = 0.15 
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        for pt in points_list:
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = 0.1
            marker.points.append(p)
        self.pub_mpc_ref.publish(marker)

    def publish_full_waypoint(self):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05 
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        for point in self.waypoints:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.0
            marker.points.append(p)
        if len(self.waypoints) > 0:
            p_first = Point()
            p_first.x = float(self.waypoints[0][0])
            p_first.y = float(self.waypoints[0][1])
            p_first.z = 0.0
            marker.points.append(p_first)
        marker_array.markers.append(marker)
        self.pub_marker_path.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = SimKinematicMPCNode()
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if "node" in locals():
            node.pub_drive.publish(AckermannDriveStamped()) 
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()