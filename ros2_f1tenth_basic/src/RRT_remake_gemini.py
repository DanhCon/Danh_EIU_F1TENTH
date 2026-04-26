#!/usr/bin/env python3

"""
INTEGRATED NODE: RRT* Planner + Nudging + Adaptive Pure Pursuit (Hybrid Logic)
Status: FINAL FIXED (Added check_collision)
"""

import numpy as np
from numpy import linalg as LA
from scipy import ndimage
import math
import csv
import os
import time
from copy import deepcopy

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
# Thêm dòng này vào khu vực import
# Thay đổi dòng import message geometry
from geometry_msgs.msg import PointStamped, PoseStamped # PointStamped cho vật cản, PoseStamped cho xóa

# --- ROS2 Messages ---
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, PoseStamped, PointStamped, Pose
from nav_msgs.msg import Odometry, OccupancyGrid
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray

# --- TF2 & Transformations ---
from tf2_ros import Buffer, TransformListener, TransformException
from tf_transformations import euler_from_quaternion

# --- CONSTANTS ---
SIMULATION = True
DEBUG = True
INVALID_INDEX = -1

# =============================================================================
# PART 1: HELPER CLASSES
# =============================================================================

class TreeNode(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.parent = None
        self.cost = None 
        self.is_root = False

class OccupancyGridManager:
    def __init__(self, x_bounds, y_bounds, cell_size, obstacle_inflation_radius, publisher, laser_frame):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.cell_size = cell_size
        self.publisher = publisher
        self.obstacle_inflation_radius = obstacle_inflation_radius
        self.laser_frame = laser_frame
        
        # Tạo lưới
        num_arrays = int((x_bounds[1] - x_bounds[0]) / cell_size) + 1
        length = int((y_bounds[1] - y_bounds[0]) / cell_size) + 1
        self.occupancy_grid = np.zeros((num_arrays, length), dtype=np.int8)

    def compute_index_from_coordinates(self, x, y):
        i = np.int32((x - self.x_bounds[0]) / self.cell_size)
        # Đảo trục i (theo logic code cũ)
        i = self.occupancy_grid.shape[0] - i - 1
        
        j = np.int32((y - self.y_bounds[0]) / self.cell_size)
        # Đảo trục j
        j = self.occupancy_grid.shape[1] - j - 1
        
        # Check bounds scalar/array
        if isinstance(i, np.ndarray):
            mask = (i >= 0) & (i < self.occupancy_grid.shape[0]) & \
                   (j >= 0) & (j < self.occupancy_grid.shape[1])
            i[~mask] = INVALID_INDEX
            j[~mask] = INVALID_INDEX
        else:
            if i < 0 or i >= self.occupancy_grid.shape[0] or \
               j < 0 or j >= self.occupancy_grid.shape[1]:
                return INVALID_INDEX, INVALID_INDEX
        return i, j

    def check_line_collision(self, x1, y1, x2, y2):
        grid_x1, grid_y1 = self.compute_index_from_coordinates(x1, y1)
        grid_x2, grid_y2 = self.compute_index_from_coordinates(x2, y2)
        
        if grid_x1 == INVALID_INDEX or grid_y1 == INVALID_INDEX or \
           grid_x2 == INVALID_INDEX or grid_y2 == INVALID_INDEX:
            return True # Coi như va chạm nếu ra ngoài map

        points = self.bresenham(grid_x1, grid_y1, grid_x2, grid_y2)
        for point in points:
            i, j = point
            if self.occupancy_grid[i, j] >0:
                return True
        return False

    def bresenham(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        is_steep = abs(dy) > abs(dx)
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
        dx = x2 - x1
        dy = y2 - y1
        error = dx // 2
        ystep = 1 if y1 < y2 else -1
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
        if swapped:
            points.reverse()
        return points

    def populate(self, scan_msg):
        """ Vectorized Population """
        self.occupancy_grid.fill(0) # Reset lưới

        ranges = np.array(scan_msg.ranges)
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        
        # Chỉ lấy điểm hợp lệ trong tầm map
        max_dist = self.x_bounds[1] * 1.5 
        valid_indices = np.where((ranges < max_dist) & (ranges > 0.05))[0]
        if len(valid_indices) == 0: return

        valid_ranges = ranges[valid_indices]
        valid_angles = angle_min + valid_indices * angle_increment

        xs = valid_ranges * np.cos(valid_angles)
        ys = valid_ranges * np.sin(valid_angles)

        grid_rows = self.occupancy_grid.shape[0]
        grid_cols = self.occupancy_grid.shape[1]

        # Convert to index directly
        i_raw = ((xs - self.x_bounds[0]) / self.cell_size).astype(int)
        j_raw = ((ys - self.y_bounds[0]) / self.cell_size).astype(int)
        
        i_indices = grid_rows - 1 - i_raw
        j_indices = grid_cols - 1 - j_raw

        # Clip mask
        mask = (i_indices >= 0) & (i_indices < grid_rows) & \
               (j_indices >= 0) & (j_indices < grid_cols)
        
        self.occupancy_grid[i_indices[mask], j_indices[mask]] = 1

        # Inflation
        inf_r = int(self.obstacle_inflation_radius / self.cell_size)
        if inf_r > 0:
            self.occupancy_grid = ndimage.binary_dilation(
                self.occupancy_grid, iterations=inf_r
            ).astype(np.int8)

        # Safety clear around robot (0,0)
        c_i, c_j = self.compute_index_from_coordinates(0, 0)
        safe_r = int(0.3 / self.cell_size)
        if c_i != INVALID_INDEX:
            r_min = max(0, c_i - safe_r); r_max = min(grid_rows, c_i + safe_r)
            c_min = max(0, c_j - safe_r); c_max = min(grid_cols, c_j + safe_r)
            self.occupancy_grid[r_min:r_max, c_min:c_max] = 0
    def add_virtual_obstacles(self, obstacle_list_car_frame):
        """
        Vẽ vật cản ảo vào lưới (sau khi đã populate laser thật).
        obstacle_list_car_frame: List các tọa độ [(x, y), ...] so với xe.
        """
        # [FIX] Tạo một lưới tạm để vẽ vật cản
        temp_grid = np.zeros_like(self.occupancy_grid, dtype=np.int8)
        
        # Bán kính vật cản ảo GỐC (trước khi phình)
        radius = 0.15  # 15cm - kích thước thật của vật cản
        radius_cells = int(radius / self.cell_size)

        for (obs_x, obs_y) in obstacle_list_car_frame:
            # Đổi tọa độ mét sang index lưới
            c_i, c_j = self.compute_index_from_coordinates(obs_x, obs_y)
            
            if c_i == INVALID_INDEX: 
                continue

            # Vẽ hình tròn vào lưới tạm
            r_min = max(0, c_i - radius_cells)
            r_max = min(temp_grid.shape[0], c_i + radius_cells + 1)
            c_min = max(0, c_j - radius_cells)
            c_max = min(temp_grid.shape[1], c_j + radius_cells + 1)

            # Vẽ hình tròn thay vì hình vuông
            for i in range(r_min, r_max):
                for j in range(c_min, c_max):
                    dist = np.sqrt((i - c_i)**2 + (j - c_j)**2)
                    if dist <= radius_cells:
                        temp_grid[i, j] = 1

        # [QUAN TRỌNG] Áp dụng inflation giống như laser scan
        inf_r = int(self.obstacle_inflation_radius / self.cell_size)
        if inf_r > 0:
            temp_grid = ndimage.binary_dilation(
                temp_grid, iterations=inf_r
            ).astype(np.int8)
        
        # Gộp vào lưới chính (OR logic)
        self.occupancy_grid = np.maximum(self.occupancy_grid, temp_grid)

    def publish_for_vis(self):
        msg = OccupancyGrid()
        msg.header.frame_id = self.laser_frame
        msg.header.stamp = self.publisher.get_clock().now().to_msg()  # Fix timestamp
        msg.info.width = self.occupancy_grid.shape[0]
        msg.info.height = self.occupancy_grid.shape[1]
        msg.info.resolution = self.cell_size
        msg.info.origin.position.x = self.x_bounds[0]
        msg.info.origin.position.y = self.y_bounds[0]
        msg.info.origin.orientation.w = 1.0
        
        # [QUAN TRỌNG] Dùng astype(int8) trực tiếp, không nhân 100
        rotated = np.rot90(self.occupancy_grid, k=1)
        flipped = np.fliplr(rotated)
        msg.data = (flipped * 100).astype(np.int8).flatten().tolist()
        
        self.publisher.publish(msg)

# =============================================================================
# PART 2: MAIN INTEGRATED NODE
# =============================================================================

class IntegratedRRTPurePursuit(Node):
    def __init__(self):
        super().__init__('integrated_rrt_pp_node')
        self.get_logger().info("Khoi tao FINAL NODE: Safety + Hybrid Lookahead + Smooth Path")
        
        # Thêm vào trong __init__
        self.obs_vis_pub_ = self.create_publisher(MarkerArray, '/rrt/obstacles', 10)

        # --- 1. TUNED PARAMETERS ---
        self.L = 0.33    
        self.kq = 1.0     # Gain vừa phải
        
        # Hybrid Lookahead Params
        self.min_lookahead = 0.4   # Nhìn gần khi tracking (bám sát)
        self.max_lookahead = 1.2   
        self.lookahead_time = 0.6  
        self.rrt_tracking_distance = 1.2 # Nhìn xa khi né RRT (để cua mượt)

        # Speed Profile (Giảm tốc để an toàn)
        self.MAX_SPEED = 1.3
        self.MIN_SPEED = 0.9 
        self.MAX_ANGLE = 0.35
        self.slope = (self.MIN_SPEED - self.MAX_SPEED) / self.MAX_ANGLE
        self.start_index = None

        # RRT Params
        self.declare_parameter('lookahead_rrt', 3.0) # Tìm đường xa 2.5m
        self.declare_parameter('max_steer_distance', 0.5)
        self.declare_parameter('rrt_delay_counter', 1) # Delay ít để phản ứng nhanh
        self.declare_parameter('cell_size', 0.2)      # 5cm - LƯỚI MỊN
        self.declare_parameter('goal_bias', 0.15)
        self.declare_parameter('goal_close_enough', 0.2)
        self.declare_parameter('obstacle_inflation_radius', 0.2) # 15cm phình to
        self.declare_parameter('num_rrt_points', 100) # Tăng mẫu vì lưới mịn
        self.declare_parameter('neighborhood_radius', 0.8)

        self.rrt_lookahead = self.get_parameter('lookahead_rrt').value
        self.max_steer_distance = self.get_parameter('max_steer_distance').value
        self.rrt_delay_counter_limit = self.get_parameter('rrt_delay_counter').value
        self.cell_size = self.get_parameter('cell_size').value
        self.goal_bias = self.get_parameter('goal_bias').value
        self.goal_close_enough = self.get_parameter('goal_close_enough').value
        self.obstacle_inflation_radius = self.get_parameter('obstacle_inflation_radius').value
        self.num_rrt_points = self.get_parameter('num_rrt_points').value
        self.neighborhood_radius = self.get_parameter('neighborhood_radius').value

        # --- TOPICS ---
        self.laser_frame = "laser"
        self.map_frame = "map"
        self.car_frame = "base_link"
        self.pose_sub_ = self.create_subscription(Odometry, "/odom", self.odom_callback, 1)
        self.scan_sub_ = self.create_subscription(LaserScan, "/scan", self.scan_callback, 1)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        # Vis
        self.goal_vis_pub_ = self.create_publisher(Marker, '/rrt/goal', 10)
        self.grid_vis_pub_ = self.create_publisher(OccupancyGrid, '/rrt/grid', 10)
        self.path_vis_pub_ = self.create_publisher(Marker, '/rrt/path', 10)
        self.tree_vis_pub_ = self.create_publisher(MarkerArray, '/rrt/tree', 10)
        self.waypoint_vis_pub_ = self.create_publisher(MarkerArray, "/publish_duong_di", 10)
        self.current_pos_pub_ = self.create_publisher(Marker, "/publish_vi_tri_hien_tai", 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- GRID INITIALIZATION ---
        # Mở rộng bản đồ để bao trọn điểm đích
        self.grid_bounds_x = (0.0, self.rrt_lookahead * 1.5) 
        self.grid_bounds_y = (-self.rrt_lookahead, self.rrt_lookahead) 
        self.occupancy_grid = OccupancyGridManager(self.grid_bounds_x,
                                                   self.grid_bounds_y,
                                                   self.cell_size, 
                                                   self.obstacle_inflation_radius, 
                                                   self.grid_vis_pub_,
                                                   self.laser_frame)
        
        self.tree = []
        self.grid_formed = False
        self.current_rrt_delay_counter = 0
        self.current_local_path = None # Bộ nhớ đường đi
        self.vis_counter = 0
        self.obstacle_sub = self.create_subscription(
            PointStamped, 
            '/clicked_point', 
            self.add_obstacle_callback, 
            10
        )
        
        # Nút "2D Nav Goal" (mũi tên hồng) vẫn dùng để XÓA vật cản (OK vì ta ko chạy Nav2 tự động)
        self.clear_obs_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.clear_obstacles_callback,
            10
        )

        self.virtual_obstacles_map_frame = []

        # Load CSV
        self.waypoints = []
        csv_path = "/home/danh/ros2_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        if os.path.exists(csv_path):
            self.load_waypoints(csv_path)
            self.publish_duong_di()
        else:
            self.get_logger().error(f"Khong tim thay file CSV: {csv_path}")

    # =========================================================================
    # PART 3: LOGIC
    # =========================================================================
    # --- CÁC HÀM MỚI CHO VẬT CẢN ẢO ---
    def publish_for_vis(self, current_time): 
        msg = OccupancyGrid()
        
        # Gán thời gian nhận được vào header
        msg.header.stamp = current_time 
        
        msg.header.frame_id = self.laser_frame
        msg.info.width = self.occupancy_grid.shape[0]
        # ... (các dòng bên dưới giữ nguyên) ...
        
        self.publisher.publish(msg)
    def visualize_obstacles(self):
        # Nếu không có vật cản, gửi một lệnh xóa sạch rồi return
        if not self.virtual_obstacles_map_frame:
            marker_array = MarkerArray()
            delete_marker = Marker()
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
            self.obs_vis_pub_.publish(marker_array)
            return

        marker_array = MarkerArray()
        
        # 1. Tạo Marker xóa tất cả cái cũ trước (Reset)
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # 2. Tạo Marker mới
        for i, (x, y) in enumerate(self.virtual_obstacles_map_frame):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            
            # [QUAN TRỌNG] Namespace giúp phân biệt các nhóm marker
            marker.ns = "obstacles" 
            
            # [QUAN TRỌNG] ID phải khác nhau trong cùng 1 lần gửi
            marker.id = i 
            
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.7; marker.scale.y = 0.7; marker.scale.z = 0.7
            marker.color.a = 1.0; marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.1
            
            marker_array.markers.append(marker)
        
        self.obs_vis_pub_.publish(marker_array)
    
    def add_obstacle_callback(self, msg):
        # PointStamped có cấu trúc msg.point.x
        x = msg.point.x
        y = msg.point.y
        
        # Lưu vào danh sách (đây là tọa độ Map Frame vì bạn click trên RViz đang set Fixed Frame = map)
        self.virtual_obstacles_map_frame.append((x, y))
        self.get_logger().info(f"Đã THẢ VẬT CẢN tại: ({x:.2f}, {y:.2f})")
        self.visualize_obstacles()

    def clear_obstacles_callback(self, msg):
        self.virtual_obstacles_map_frame.clear()
        self.get_logger().info("Da XOA sach vat can ao!")
        self.visualize_obstacles()

    # --- SỬA LẠI HÀM scan_callback ---
    def scan_callback(self, scan_msg: LaserScan):
    # 1. Nạp laser thật
        self.occupancy_grid.populate(scan_msg)

        # 2. Thêm vật cản ảo
        if self.virtual_obstacles_map_frame:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.map_frame, self.car_frame, rclpy.time.Time())
                
                current_car_pose = Pose()
                current_car_pose.position.x = t.transform.translation.x
                current_car_pose.position.y = t.transform.translation.y
                current_car_pose.orientation = t.transform.rotation

                obs_local_list = []
                for (mx, my) in self.virtual_obstacles_map_frame:
                    p_local = self.transform_point_to_car_frame([mx, my], current_car_pose)
                    
                    # Chỉ thêm nếu trong phạm vi grid
                    if (0 < p_local[0] < self.grid_bounds_x[1]) and \
                    (self.grid_bounds_y[0] < p_local[1] < self.grid_bounds_y[1]):
                        obs_local_list.append(p_local)
                
                if obs_local_list:
                    self.occupancy_grid.add_virtual_obstacles(obs_local_list)

            except TransformException:
                pass

        self.grid_formed = True
        
        # 3. Visualize THƯA HƠN để giảm lag
        # if DEBUG:
        #     self.vis_counter += 1
        #     if self.vis_counter % 20 == 0:  # Tăng từ 5 → 10
        #         now = self.get_clock().now().to_msg()
        #         self.occupancy_grid.publish_for_vis()
        #         self.vis_counter = 0


    def transform_point_to_car_frame(self, point, car_pose_msg):
        cx = car_pose_msg.position.x
        cy = car_pose_msg.position.y
        q = [car_pose_msg.orientation.x, car_pose_msg.orientation.y, 
             car_pose_msg.orientation.z, car_pose_msg.orientation.w]
        yaw = euler_from_quaternion(q)[2]
        dx = point[0] - cx
        dy = point[1] - cy
        x_car = dx * np.cos(yaw) + dy * np.sin(yaw)
        y_car = -dx * np.sin(yaw) + dy * np.cos(yaw)
        return np.array([x_car, y_car])

    def transform_point_to_map_frame(self, point, car_pose_msg):
        cx = car_pose_msg.position.x
        cy = car_pose_msg.position.y
        q = [car_pose_msg.orientation.x, car_pose_msg.orientation.y, 
             car_pose_msg.orientation.z, car_pose_msg.orientation.w]
        yaw = euler_from_quaternion(q)[2]
        x_map = point[0] * np.cos(yaw) - point[1] * np.sin(yaw) + cx
        y_map = point[0] * np.sin(yaw) + point[1] * np.cos(yaw) + cy
        return np.array([x_map, y_map])

    def try_nudging_goal(self, original_goal_car, car_pose):
        """ LÁCH NHẸ (NUDGING) """
        # Khôi phục mảng shifts để thực sự lách
        shifts = np.array([0.0, 2.0, -2.0])
        valid_goal_car = None

        for shift in shifts:
            candidate_x = original_goal_car[0]
            candidate_y = original_goal_car[1] + shift 
            
            # Check đường thẳng
            if not self.occupancy_grid.check_line_collision(0, 0, candidate_x, candidate_y):
                valid_goal_car = np.array([candidate_x, candidate_y])
                if DEBUG and shift != 0.0:
                    self.get_logger().info(f"Nudging: {shift}m", throttle_duration_sec=0.5)
                break
        
        if valid_goal_car is not None:
            return self.transform_point_to_map_frame(valid_goal_car, car_pose)
        return None

    def is_current_path_safe(self, path):
        """ PATH COMMITMENT: Kiểm tra đường cũ """
        if path is None or len(path) < 2: return False
        for i in range(len(path) - 1):
            n1 = path[i]; n2 = path[i+1]
            if self.occupancy_grid.check_line_collision(n1.x, n1.y, n2.x, n2.y):
                return False 
        return True

    def odom_callback(self, pose_msg: Odometry):
        if not self.grid_formed or not self.waypoints: return

        # --- TF LOOKUP ---
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.car_frame, rclpy.time.Time(), Duration(seconds=0.05)
            )
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            car_pose = Pose()
            car_pose.position.x = robot_x
            car_pose.position.y = robot_y
            car_pose.position.z = 0.0
            car_pose.orientation = transform.transform.rotation
            
            self.publish_vi_tri_hien_tai(robot_x, robot_y)
        except TransformException: return

        # 1. Tính toán Ld Bình thường (Tracking)
        vx = pose_msg.twist.twist.linear.x; vy = pose_msg.twist.twist.linear.y
        current_speed = math.hypot(vx, vy)
        normal_Ld = np.clip(current_speed * self.lookahead_time, self.min_lookahead, self.max_lookahead)
        
        # Mặc định dùng normal_Ld
        current_active_Ld = normal_Ld 

        # 2. Tìm điểm đích XA để check vật cản (Detection Zone)
        check_dist = max(self.rrt_lookahead, 1.5)
        global_goal_map = self.get_diem_lookahead(robot_x, robot_y, lookahead_dist=check_dist)
        if global_goal_map is None: return
        
        global_goal_car = self.transform_point_to_car_frame(global_goal_map, car_pose)
        
        # Clamp goal
        max_x_allowed = self.grid_bounds_x[1] * 0.95
        global_goal_car[0] = np.clip(global_goal_car[0], 0.0, max_x_allowed)

        # ==========================================
        # HYBRID LOGIC STATE MACHINE
        # ==========================================
        target_point_to_drive = None
        is_avoidance_mode = False

        # State 1: Đang bám RRT (Né)
        if self.current_local_path is not None:
            if self.is_current_path_safe(self.current_local_path):
                is_avoidance_mode = True
                # Check nếu gần hết đường RRT thì xóa
                dist_to_end = math.hypot(self.current_local_path[-1].x, self.current_local_path[-1].y)
                if dist_to_end < 0.3: self.current_local_path = None
            else:
                self.current_local_path = None # Đường RRT bị chặn -> Xóa

        # State 2: Chưa có RRT, check vật cản trên đường thẳng
        if not is_avoidance_mode:
            # Check va chạm tới điểm đích XA
            if self.occupancy_grid.check_line_collision(0, 0, global_goal_car[0], global_goal_car[1]):
                target_speed = 1.0
                # Tắc đường -> Thử Nudging trước
                nudged_map = self.try_nudging_goal(global_goal_car, car_pose)
                
                if nudged_map is not None:
                    target_point_to_drive = nudged_map # Lách nhẹ thành công
                    # Lưu ý: Lách nhẹ coi như là Tracking mode nhưng điểm lệch đi
                else:
                    # Lách không được -> RRT*
                    self.get_logger().warn("BLOCKED! RRT Planning...", throttle_duration_sec=1.0)
                    path_nodes = self.rrt_star(global_goal_car)
                    if path_nodes and len(path_nodes) > 1:
                        self.current_local_path = path_nodes
                        is_avoidance_mode = True
                        if DEBUG:
                            self.visualize_path(path_nodes)
                            self.visualize_tree()

        # === CHỌN TARGET THEO MODE ===
        if is_avoidance_mode and self.current_local_path:
            # Mode Né: Nhìn xa hơn để cua mượt
            current_active_Ld = self.rrt_tracking_distance
            best_node = self.find_waypoint_to_track_rrt(self.current_local_path, current_active_Ld)
            target_point_to_drive = self.transform_point_to_map_frame([best_node.x, best_node.y], car_pose)
        
        elif target_point_to_drive is None:
            # Mode Đua: Lấy lại điểm CSV ở cự ly GẦN (normal_Ld)
            short_goal_map = self.get_diem_lookahead(robot_x, robot_y, lookahead_dist=normal_Ld)
            if short_goal_map is not None:
                target_point_to_drive = short_goal_map
            current_active_Ld = normal_Ld

        # Cập nhật Ld để hàm drive dùng
        self.Ld = current_active_Ld
        if target_point_to_drive is not None:
            self.drive_pure_pursuit_adaptive(target_point_to_drive, robot_x, robot_y, car_pose)

    def drive_pure_pursuit_adaptive(self, target_map, robot_x, robot_y, car_pose_msg):
        # --- 1. SAFETY LAYER: PHANH KHẨN CẤP ---
        # Check vùng 0.5m trước mặt
        grid_x, _ = self.occupancy_grid.compute_index_from_coordinates(0, 0)
        if grid_x != INVALID_INDEX:
            if self.occupancy_grid.check_line_collision(0.1, 0.0, 0.3, 0.0):
                self.get_logger().warn("EMERGENCY BRAKE!", throttle_duration_sec=0.2)
                drive_msg = AckermannDriveStamped()
                drive_msg.header.stamp = self.get_clock().now().to_msg()
                drive_msg.header.frame_id = self.car_frame
                drive_msg.drive.speed = 0.0
                self.drive_pub_.publish(drive_msg)
                return 

        # --- PURE PURSUIT ---
        target_local = self.transform_point_to_car_frame(target_map, car_pose_msg)
        x_local, y_local = target_local[0], target_local[1]
        actual_dist = math.hypot(x_local, y_local)
        
        if actual_dist < 0.01: return

        # Rescale logic
        effective_Ld = self.Ld
        if actual_dist > effective_Ld:
            scale = effective_Ld / actual_dist
            x_local *= scale; y_local *= scale
            Ld_square = effective_Ld**2
        else:
            Ld_square = actual_dist**2

        curvature = 2.0 * y_local / Ld_square
        steering_angle = math.atan(curvature * self.L) * self.kq
        steering_angle = np.clip(steering_angle, -self.MAX_ANGLE, self.MAX_ANGLE)

        # Smart Speed: Cua gắt -> Giảm tốc
        if abs(steering_angle) > 0.2: # > 11 độ
             speed = self.MIN_SPEED
        else:
             speed = self.MAX_SPEED

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = self.car_frame
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub_.publish(drive_msg)

    # ... RRT METHODS ...
    
    def check_collision(self, n1, n2):
        """ Hàm kiểm tra va chạm giữa 2 node """
        return self.occupancy_grid.check_line_collision(n1.x, n1.y, n2.x, n2.y)

    def rrt_star(self, goal):
        start_node = TreeNode()
        start_node.x, start_node.y = 0.0, 0.0
        start_node.cost = 0; start_node.is_root = True
        self.tree = [start_node]
        goal_with_min_cost = None
        for _ in range(self.num_rrt_points):
            sampled_point = self.sample(goal)
            nearest_node = self.nearest(self.tree, sampled_point)
            new_node = self.steer(nearest_node, sampled_point)
            if self.check_collision(nearest_node, new_node): continue
            new_node.cost, new_node.parent = self.calc_cost_new_node(self.tree, new_node)
            if new_node.parent is None: continue
            self.tree.append(new_node)
            self.rewire(self.tree, new_node)
            if self.is_goal(new_node, goal[0], goal[1]):
                if goal_with_min_cost is None or new_node.cost < goal_with_min_cost.cost:
                    goal_with_min_cost = new_node
        if goal_with_min_cost: return self.find_path(self.tree, goal_with_min_cost)
        return None

    def sample(self, goal):
        if np.random.uniform() < self.goal_bias: return (goal[0], goal[1])
        mean = np.array([goal[0], goal[1]]) / 2
        std_dev = LA.norm(np.array([goal[0], goal[1]])) / 2
        x = np.random.normal(mean[0], std_dev)
        y = np.random.normal(mean[1], std_dev/2) 
        x = np.clip(x, 0.0, self.grid_bounds_x[1]) 
        y = np.clip(y, self.grid_bounds_y[0], self.grid_bounds_y[1])
        return (x, y)

    def nearest(self, tree, sampled_point):
        nearest_node = None
        nearest_dist = float('inf')
        arr_sample = np.array(sampled_point)
        for node in tree:
            dist = LA.norm(np.array([node.x, node.y]) - arr_sample)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_node = node
        return nearest_node

    def steer(self, nearest_node, sampled_point):
        vec = np.array(sampled_point) - np.array([nearest_node.x, nearest_node.y])
        dist = LA.norm(vec)

        # [FIX] Kiểm tra nếu khoảng cách quá nhỏ (trùng điểm) để tránh chia cho 0
        if dist < 1e-6: 
            new_node = TreeNode()
            new_node.x = nearest_node.x
            new_node.y = nearest_node.y
            new_node.parent = nearest_node
            return new_node

        # Chỉ chia khi dist > 0
        vec = vec / dist 
        
        new_node = TreeNode()
        if dist > self.max_steer_distance:
            new_node.x = nearest_node.x + self.max_steer_distance * vec[0]
            new_node.y = nearest_node.y + self.max_steer_distance * vec[1]
        else:
            new_node.x = sampled_point[0]
            new_node.y = sampled_point[1]
            
        new_node.parent = nearest_node
        return new_node

    def is_goal(self, node, gx, gy):
        dist = LA.norm(np.array([node.x, node.y]) - np.array([gx, gy]))
        return dist <= self.goal_close_enough

    def find_path(self, tree, node):
        path = []
        curr = node
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        path.reverse()
        return path

    def calc_cost_new_node(self, tree, node):
        if node.is_root: return 0, None
        neighborhood = self.near(tree, node)
        min_cost = float('inf')
        parent = None
        for n in neighborhood:
            cost = n.cost + self.line_cost(n, node)
            if cost < min_cost:
                if not self.check_collision(n, node):
                    min_cost = cost
                    parent = n
        return min_cost, parent

    def rewire(self, tree, node):
        neighborhood = self.near(tree, node)
        for n in neighborhood:
            new_cost = node.cost + self.line_cost(node, n)
            if new_cost < n.cost:
                if not self.check_collision(node, n):
                    n.parent = node
                    n.cost = new_cost

    def near(self, tree, node):
        neighborhood = []
        for n in tree:
            if LA.norm(np.array([n.x, n.y]) - np.array([node.x, node.y])) <= self.neighborhood_radius:
                neighborhood.append(n)
        return neighborhood

    def line_cost(self, n1, n2):
        return LA.norm(np.array([n1.x, n1.y]) - np.array([n2.x, n2.y]))

    def find_waypoint_to_track_rrt(self, path, Ld):
        best_node = path[-1] 
        for node in path:
            dist = math.hypot(node.x, node.y)
            if dist >= Ld:
                best_node = node
                break
        return best_node

    def load_waypoints(self, filename):
        raw_waypoints = []
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row: continue
                    line_data = row
                    if len(row) == 1 and isinstance(row[0], str):
                         line_data = row[0].split()
                    try:
                        x = float(line_data[0])
                        y = float(line_data[1])
                        raw_waypoints.append([x, y])
                    except ValueError: continue
            self.waypoints = self.smooth_path(raw_waypoints)
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")
        except Exception as e:
            self.get_logger().error(f"Error loading CSV: {e}")

    def smooth_path(self, path, weight_data=0.5, weight_smooth=0.2, tolerance=0.00001):
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

    def dist(self, p1, p2):
        return math.sqrt((p1[0] -  p2[0])**2 + (p1[1] - p2[1])**2)

    def find_giao_diem_voi_vong_tron_ahead(self, p1, p2, robot_pos, r):
        d = p2 - p1
        f = p1 - robot_pos
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - r**2
        delta = b**2 - 4*a*c
        if delta < 0: return None
        sqrt_dis = math.sqrt(delta)
        t1 = (-b - sqrt_dis) / (2*a)
        t2 = (-b + sqrt_dis) / (2*a)
        if 0 <= t2 <= 1: return p1 + t2*d
        elif 0 <= t1 <= 1: return p1 + t1*d
        return None

    def get_diem_lookahead(self, robot_x, robot_y, lookahead_dist):
        robot_pos = np.array([robot_x, robot_y])
        min_dist = float('inf')
        num_waypoints = len(self.waypoints)
        if self.start_index is None:
            for i, point in enumerate(self.waypoints):
                d = self.dist([robot_x, robot_y], point)
                if d < min_dist:
                    min_dist = d
                    self.start_index = i
        else:
            curr_dist = self.dist([robot_x, robot_y], self.waypoints[self.start_index])
            for i in range (40):
                next_idx = (self.start_index + 1) % num_waypoints
                next_dist = self.dist([robot_x, robot_y], self.waypoints[next_idx])
                if next_dist < curr_dist:
                    self.start_index = next_idx
                    curr_dist = next_dist
                else:
                    break
        nearest_idx = self.start_index
        lookahead_point = None
        for i in range(40):
            idx_start = (nearest_idx + i) % len(self.waypoints)
            idx_end = (nearest_idx + i + 1) % len(self.waypoints)
            p1 = np.array(self.waypoints[idx_start])
            p2 = np.array(self.waypoints[idx_end])
            intersection = self.find_giao_diem_voi_vong_tron_ahead(p1, p2, robot_pos, lookahead_dist)
            if intersection is not None:
                lookahead_point = intersection
                break
        if lookahead_point is None:
             fallback_idx = (nearest_idx + 5) % len(self.waypoints)
             lookahead_point = np.array(self.waypoints[fallback_idx])
        return lookahead_point

    def visualize_path(self, path):
        marker = Marker()
        marker.header.frame_id = self.laser_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 1.0; marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0 
        for node in path:
            marker.points.append(Point(x=node.x, y=node.y, z=0.0))
        self.path_vis_pub_.publish(marker)

    def visualize_tree(self):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = self.laser_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05; marker.scale.y = 0.05
        marker.color.a = 1.0; marker.color.b = 1.0
        marker.id = 0
        for node in self.tree:
            marker.points.append(Point(x=node.x, y=node.y, z=0.0))
        marker_array.markers.append(marker)
        marker_line = Marker()
        marker_line.header.frame_id = self.laser_frame
        marker_line.header.stamp = self.get_clock().now().to_msg()
        marker_line.type = Marker.LINE_LIST
        marker_line.action = Marker.ADD
        marker_line.scale.x = 0.02
        marker_line.color.a = 0.5; marker_line.color.b = 1.0
        marker_line.id = 1
        for node in self.tree:
            if node.parent:
                marker_line.points.append(Point(x=node.x, y=node.y, z=0.0))
                marker_line.points.append(Point(x=node.parent.x, y=node.parent.y, z=0.0))
        marker_array.markers.append(marker_line)
        self.tree_vis_pub_.publish(marker_array)

    def visualize_goal(self, point_car_frame):
        marker = Marker()
        marker.header.frame_id = self.laser_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.2; marker.scale.y = 0.2; marker.scale.z = 0.2
        marker.color.a = 1.0; marker.color.g = 1.0 
        marker.pose.position.x = point_car_frame[0]
        marker.pose.position.y = point_car_frame[1]
        self.goal_vis_pub_.publish(marker)

    def publish_vi_tri_hien_tai(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3
        marker.color.a = 1.0; marker.color.r = 1.0; marker.color.g = 1.0
        marker.pose.position.x = x; marker.pose.position.y = y
        self.current_pos_pub_.publish(marker)

    def publish_duong_di(self):
        marker_array = MarkerArray()
        for i, point in enumerate(self.waypoints):
            if i % 2 == 0: 
                marker = Marker()
                marker.header.frame_id = "map"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.scale.x = 0.1; marker.scale.y = 0.1; marker.scale.z = 0.1
                marker.color.a = 0.5; marker.color.r = 0.0; marker.color.g = 1.0
                marker.pose.position.x = point[0]; marker.pose.position.y = point[1]
                marker_array.markers.append(marker)
        self.waypoint_vis_pub_.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = IntegratedRRTPurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()