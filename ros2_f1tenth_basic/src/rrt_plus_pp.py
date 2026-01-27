#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import csv
import numpy as np
import os
import time
import random
from copy import deepcopy

# --- IMPORT MSG ---
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
# QUAN TRỌNG: Import Vector3 để sửa lỗi Visualization
from geometry_msgs.msg import PointStamped, Point, Vector3 
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs
from rclpy.duration import Duration

# ==========================================
# 1. CLASS RRT (LOCAL PLANNER)
# ==========================================
class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=0.5, goal_sample_rate=10, max_iter=150):
        self.start = RRTNode(start[0], start[1])
        self.goal = RRTNode(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis    # Khoảng cách mỗi bước mở rộng
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # Sampling
            if random.randint(0, 100) > self.goal_sample_rate:
                rnd_point = [random.uniform(self.min_rand, self.max_rand),
                             random.uniform(self.min_rand, self.max_rand)]
            else:
                rnd_point = [self.goal.x, self.goal.y]

            # Nearest Node
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_point)
            nearest_node = self.node_list[nearest_ind]

            # Steer
            theta = math.atan2(rnd_point[1] - nearest_node.y, rnd_point[0] - nearest_node.x)
            new_node = RRTNode(nearest_node.x + self.expand_dis * math.cos(theta),
                               nearest_node.y + self.expand_dis * math.sin(theta))
            new_node.parent = nearest_node

            # Collision Check
            if not self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            # Check to Goal
            dx = new_node.x - self.goal.x
            dy = new_node.y - self.goal.y
            d = math.hypot(dx, dy)
            
            if d <= self.expand_dis:
                final_node = RRTNode(self.goal.x, self.goal.y)
                final_node.parent = new_node
                return self.generate_course(final_node)

        return None

    def get_nearest_node_index(self, node_list, rnd_point):
        dlist = [(node.x - rnd_point[0])**2 + (node.y - rnd_point[1])**2 for node in node_list]
        return dlist.index(min(dlist))

    def check_collision(self, node, obstacle_list):
        if obstacle_list is None: return False
        SAFE_MARGIN = 0.4  # Bán kính an toàn (mét)
        for (ox, oy, size) in obstacle_list:
            dx = ox - node.x
            dy = oy - node.y
            d = math.hypot(dx, dy)
            if d <= (size + SAFE_MARGIN):
                return True
        return False

    def generate_course(self, goal_node):
        path = [[goal_node.x, goal_node.y]]
        node = goal_node
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y])
        return path[::-1] # Đảo ngược để có từ Start -> Goal

# ==========================================
# 2. MAIN CLASS: PURE PURSUIT + RRT
# ==========================================

class PurePursuitRRT(Node):
    def __init__(self):
        super().__init__("pure_pursuit_rrt_node")
        
        # --- CẤU HÌNH XE ---
        self.L = 0.3302      # Wheelbase
        self.kq = 0.8        # Gain góc lái
        
        # --- CẤU HÌNH LOOKAHEAD ---
        self.lookahead_time = 0.8
        self.min_lookahead = 0.6
        self.max_lookahead = 3.0
        self.Ld = self.min_lookahead

        # --- CẤU HÌNH VẬN TỐC ---
        self.MAX_SPEED = 1.5
        self.MIN_SPEED = 0.8 
        self.OBSTACLE_SPEED = 0.8 # Tốc độ chậm khi đang né vật cản

        # --- FRAME ID ---
        self.car_frame = "base_link"   # Frame gắn trên xe
        self.map_frame = "map"         # Frame bản đồ
        self.laser_frame = "laser"     # Frame của cảm biến LiDAR

        # --- STATE VARIABLES ---
        self.global_waypoints = []  # Load từ CSV
        self.local_waypoints = []   # Sinh ra bởi RRT
        self.obstacles_map = []     # Vật cản tọa độ Map
        self.is_obstacle_detected = False
        self.current_pose = None    # [x, y]
        self.start_index = 0        # Index hiện tại trên global path

        # --- TF BUFFER ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- PUB/SUB ---
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        
        # VISUALIZATION
        self.pub_global_path = self.create_publisher(MarkerArray, "/viz_global_path", 10)
        self.pub_local_path = self.create_publisher(MarkerArray, "/viz_local_path", 10)
        self.pub_lookahead = self.create_publisher(Marker, "/viz_lookahead", 10)
        
        # LOAD CSV (Kiểm tra đường dẫn của bạn)
        csv_path = "/home/danh/ros2_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        if os.path.exists(csv_path):
            self.load_waypoints(csv_path)
            # Visualize đường gốc
            self.publish_path(self.global_waypoints, self.pub_global_path, [0.0, 1.0, 0.0]) # Màu Xanh
        else:
            self.get_logger().error(f"Khong tim thay CSV: {csv_path}")

        self.get_logger().info("Pure Pursuit RRT + Smoothing KHOI DONG!")

    # -----------------------------------------------------------------
    # LOGIC 1: XỬ LÝ LIDAR & PHÁT HIỆN VẬT CẢN
    # -----------------------------------------------------------------
    def scan_callback(self, msg: LaserScan):
        if self.current_pose is None: return

        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        
        # 1. Lọc vùng quan tâm (ROI): Chỉ nhìn phía trước +/- 30 độ
        num_rays = len(ranges)
        center = num_rays // 2
        window = int(math.radians(30) / angle_inc)
        
        # Lấy dữ liệu phía trước
        front_ranges = ranges[center - window : center + window]
        
        # Kiểm tra có vật cản gần (< 1.5m) không?
        valid_idx = np.where((front_ranges < 1.5) & (front_ranges > 0.1))[0]
        
        self.obstacles_map = [] # Reset list vật cản
        
        if len(valid_idx) > 5: # Nếu có cụm điểm chắn đường
            self.is_obstacle_detected = True
            
            # 2. Biến đổi tọa độ các điểm vật cản sang MAP Frame (để RRT dùng)
            try:
                # Lấy transform từ Laser -> Map
                trans = self.tf_buffer.lookup_transform(self.map_frame, self.laser_frame, rclpy.time.Time())
                
                # Downsample (Lấy mẫu thưa ra để tính cho nhanh, cứ 15 tia lấy 1)
                for i in range(0, num_rays, 15): 
                    r = ranges[i]
                    if 0.1 < r < 3.0: # Chỉ lấy vật cản trong bán kính 3m
                        angle = angle_min + i * angle_inc
                        # Polar -> Cartesian (tại khung Laser)
                        lx = r * math.cos(angle)
                        ly = r * math.sin(angle)
                        
                        # Transform -> Map
                        p_stamped = PointStamped()
                        p_stamped.header.frame_id = self.laser_frame
                        p_stamped.point.x, p_stamped.point.y = float(lx), float(ly)
                        p_out = tf2_geometry_msgs.do_transform_point(p_stamped, trans)
                        
                        # Lưu: [x, y, radius]
                        self.obstacles_map.append([p_out.point.x, p_out.point.y, 0.15])
                        
            except TransformException: pass
        else:
            self.is_obstacle_detected = False

    # -----------------------------------------------------------------
    # LOGIC 2: ĐIỀU KHIỂN & LẬP KẾ HOẠCH (MAIN LOOP)
    # -----------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        # Update vị trí xe
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        self.current_pose = [px, py]
        
        # Update tốc độ & Ld
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        spd = math.hypot(vx, vy)
        self.Ld = np.clip(spd * self.lookahead_time, self.min_lookahead, self.max_lookahead)

        # Mặc định dùng Global Path
        active_waypoints = self.global_waypoints
        target_speed = self.MAX_SPEED

        # === TRIGGER RRT KHI CÓ VẬT CẢN ===
        if self.is_obstacle_detected:
            target_speed = self.OBSTACLE_SPEED # Giảm tốc
            
            # 1. Chọn điểm đích tạm thời (Rejoin Point)
            # Chọn một điểm trên Global Path ở phía trước xe khoảng 2.5m (offset index 25)
            rejoin_idx = (self.start_index + 25) % len(self.global_waypoints)
            goal_pos = self.global_waypoints[rejoin_idx]
            
            # 2. Tạo vùng random giới hạn (Local Window)
            search_area = [px - 4.0, px + 4.0, py - 4.0, py + 4.0]
            
            # 3. Chạy RRT
            rrt = RRT(start=[px, py], goal=goal_pos, 
                      obstacle_list=self.obstacles_map, 
                      rand_area=search_area, max_iter=100)
            
            raw_path = rrt.planning()
            
            if raw_path is not None:
                # 4. LÀM MỊN ĐƯỜNG RRT (SMOOTHING)
                # weight_smooth thấp (0.3) để tránh kéo đường cong đâm vào vật cản
                self.local_waypoints = self.smooth_path(raw_path, weight_data=0.5, weight_smooth=0.3)
                
                active_waypoints = self.local_waypoints
                
                # Visualize đường đỏ (Local)
                self.publish_path(self.local_waypoints, self.pub_local_path, [1.0, 0.0, 0.0])
            else:
                self.get_logger().warn("RRT khong tim thay duong! Dung xe khan cap.")
                self.stop_vehicle()
                return
        else:
             # Nếu không có vật cản, xóa đường local viz cho đỡ rối
             empty = MarkerArray()
             # self.pub_local_path.publish(empty) # Optional

        # === PURE PURSUIT ALGORITHM ===
        # Tìm điểm Lookahead trên active_waypoints
        target_global, idx_found = self.get_diem_lookahead(px, py, active_waypoints)
        
        # Nếu đang chạy Global, cập nhật index để lần sau tìm nhanh hơn
        if not self.is_obstacle_detected:
            self.start_index = idx_found

        if target_global is None: return

        # Visualize điểm xanh (Target)
        self.publish_lookahead_marker(target_global[0], target_global[1])

        # Transform về Local Frame của xe để tính góc lái
        target_local = self.transform_to_car_frame(target_global)
        if target_local is None: return

        # Tính toán góc lái (Curvature)
        x_loc, y_loc = target_local[0], target_local[1]
        dist2 = x_loc**2 + y_loc**2
        
        if dist2 < 0.001: return

        curvature = 2.0 * y_loc / dist2
        steering = math.atan(curvature * self.L) * self.kq
        steering = np.clip(steering, -0.4, 0.4) # Giới hạn góc lái

        # Publish Drive
        drv = AckermannDriveStamped()
        drv.drive.steering_angle = steering
        drv.drive.speed = target_speed
        self.pub_drive.publish(drv)

    # -----------------------------------------------------------------
    # HÀM LÀM MỊN (SMOOTHING)
    # -----------------------------------------------------------------
    def smooth_path(self, path, weight_data=0.5, weight_smooth=0.1, tolerance=0.00001):
        """
        Làm mịn đường dẫn sử dụng Gradient Descent.
        """
        if path is None or len(path) < 3: return path

        new_path = deepcopy(path)
        change = tolerance
        
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path) - 1):
                aux_x = new_path[i][0]
                aux_y = new_path[i][1]
                
                new_path[i][0] += weight_data * (path[i][0] - new_path[i][0]) + \
                                  weight_smooth * (new_path[i-1][0] + new_path[i+1][0] - 2.0 * new_path[i][0])
                new_path[i][1] += weight_data * (path[i][1] - new_path[i][1]) + \
                                  weight_smooth * (new_path[i-1][1] + new_path[i+1][1] - 2.0 * new_path[i][1])
                
                change += abs(aux_x - new_path[i][0]) + abs(aux_y - new_path[i][1])
        return new_path

    # -----------------------------------------------------------------
    # CÁC HÀM HỖ TRỢ KHÁC
    # -----------------------------------------------------------------
    def get_diem_lookahead(self, px, py, waypoints):
        # Tìm điểm gần nhất trước
        dists = [math.hypot(p[0]-px, p[1]-py) for p in waypoints]
        nearest_idx = np.argmin(dists)
        
        # Tìm điểm Lookahead
        lookahead_pt = None
        for i in range(nearest_idx, len(waypoints)):
            d = math.hypot(waypoints[i][0]-px, waypoints[i][1]-py)
            if d > self.Ld:
                lookahead_pt = waypoints[i]
                break
        
        if lookahead_pt is None:
            lookahead_pt = waypoints[-1]
            
        return lookahead_pt, nearest_idx

    def transform_to_car_frame(self, target):
        try:
            t = self.tf_buffer.lookup_transform(self.car_frame, self.map_frame, rclpy.time.Time())
            p = PointStamped()
            p.header.frame_id = self.map_frame
            p.point.x, p.point.y = float(target[0]), float(target[1])
            p_tf = tf2_geometry_msgs.do_transform_point(p, t)
            return [p_tf.point.x, p_tf.point.y]
        except: return None

    def load_waypoints(self, filename):
        raw = []
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    try: raw.append([float(row[0]), float(row[1])])
                    except: pass
            # Làm mịn global path 1 lần lúc khởi động
            self.global_waypoints = self.smooth_path(raw, weight_data=0.5, weight_smooth=0.4)
            self.get_logger().info(f"Loaded {len(self.global_waypoints)} global waypoints.")
        except Exception as e:
            self.get_logger().error(f"Error loading CSV: {e}")

    def stop_vehicle(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        self.pub_drive.publish(msg)

    # --- SỬA LỖI Ở ĐÂY: DÙNG VECTOR3 CHO SCALE ---
    def publish_lookahead_marker(self, x, y):
        m = Marker()
        m.header.frame_id = "map"
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.scale = Vector3(x=0.3, y=0.3, z=0.3) # Sử dụng Vector3
        m.color.a = 1.0; m.color.g = 1.0 # Green
        m.pose.position.x = x
        m.pose.position.y = y
        self.pub_lookahead.publish(m)

    def publish_path(self, points, pub, color_rgb):
        ma = MarkerArray()
        for i, p in enumerate(points):
            m = Marker()
            m.header.frame_id = "map"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.scale = Vector3(x=0.1, y=0.1, z=0.1) # Sử dụng Vector3
            m.color.a = 1.0
            m.color.r, m.color.g, m.color.b = color_rgb[0], color_rgb[1], color_rgb[2]
            m.pose.position.x = p[0]
            m.pose.position.y = p[1]
            ma.markers.append(m)
        pub.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitRRT()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.stop_vehicle()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()