#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import csv
import numpy as np
import os
import time
from copy import deepcopy
from scipy import ndimage 

# --- IMPORT VISUALIZATION & MAP ---
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry, OccupancyGrid
# ----------------------------

from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs
from rclpy.duration import Duration
from geometry_msgs.msg import PointStamped

INVALID_INDEX = -1 

# ==========================================
# CLASS QUẢN LÝ BẢN ĐỒ LƯỚI (OCCUPANCY GRID)
# ==========================================
class OccupancyGridManager:
    """Fast Occupancy Grid with Virtual Obstacle Support"""
    def __init__(self, x_bounds, y_bounds, cell_size, obstacle_inflation_radius, node_obj, pub_obj, laser_frame):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.cell_size = cell_size
        self.node = node_obj       
        self.publisher = pub_obj   
        self.obstacle_inflation_radius = obstacle_inflation_radius
        self.laser_frame = laser_frame
        
        num_rows = int((x_bounds[1] - x_bounds[0]) / cell_size) + 1
        num_cols = int((y_bounds[1] - y_bounds[0]) / cell_size) + 1
        self.occupancy_grid = np.zeros((num_rows, num_cols), dtype=np.int8)

    def compute_index_from_coordinates(self, x, y):
        i = np.int32((x - self.x_bounds[0]) / self.cell_size)
        i = self.occupancy_grid.shape[0] - i - 1  
        
        j = np.int32((y - self.y_bounds[0]) / self.cell_size)
        j = self.occupancy_grid.shape[1] - j - 1  
        
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
            return True  
        
        points = self.bresenham(grid_x1, grid_y1, grid_x2, grid_y2)
        for i, j in points:
            if self.occupancy_grid[i, j] > 0:
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
        if swapped: points.reverse()
        return points

    def add_virtual_obstacles(self, obstacle_list_car_frame):
        temp_grid = np.zeros_like(self.occupancy_grid, dtype=np.int8)
        radius = 0.15 
        radius_cells = int(radius / self.cell_size)
        
        for obs_x, obs_y in obstacle_list_car_frame:
            c_i, c_j = self.compute_index_from_coordinates(obs_x, obs_y)
            if c_i == INVALID_INDEX: continue
            
            r_min = max(0, c_i - radius_cells)
            r_max = min(temp_grid.shape[0], c_i + radius_cells + 1)
            c_min = max(0, c_j - radius_cells)
            c_max = min(temp_grid.shape[1], c_j + radius_cells + 1)
            
            for i in range(r_min, r_max):
                for j in range(c_min, c_max):
                    dist = np.sqrt((i - c_i)**2 + (j - c_j)**2)
                    if dist <= radius_cells:
                        temp_grid[i, j] = 1
        
        inf_r = int(self.obstacle_inflation_radius / self.cell_size)
        if inf_r > 0:
            temp_grid = ndimage.binary_dilation(temp_grid, iterations=inf_r).astype(np.int8)
        self.occupancy_grid = np.maximum(self.occupancy_grid, temp_grid)

    def publish_for_vis(self):
        msg = OccupancyGrid()
        msg.header.frame_id = self.laser_frame
        msg.header.stamp = self.node.get_clock().now().to_msg() 
        msg.info.width = self.occupancy_grid.shape[0]
        msg.info.height = self.occupancy_grid.shape[1]
        msg.info.resolution = self.cell_size
        msg.info.origin.position.x = self.x_bounds[0]
        msg.info.origin.position.y = self.y_bounds[0]
        msg.info.origin.orientation.w = 1.0
        
        rotated = np.rot90(self.occupancy_grid, k=1)
        flipped = np.fliplr(rotated)
        msg.data = (flipped * 100).astype(np.int8).flatten().tolist()
        self.publisher.publish(msg)

# ==========================================
# NODE PURE PURSUIT ĐIỀU KHIỂN XE
# ==========================================
class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit_node")
        
        # --- CẤU HÌNH XE VÀ THUẬT TOÁN ---
        self.L = 0.39    
        self.kq = 1.0    
        self.lookahead_time = 0.4    
        self.min_lookahead = 0.4     
        self.max_lookahead = 4.0     
        self.Ld = self.min_lookahead 

        self.MAX_SPEED = 5.2
        self.MIN_SPEED = 1.2 
        self.MAX_ANGLE = 0.35
        self.slope = (self.MIN_SPEED - self.MAX_SPEED) / self.MAX_ANGLE
        self.start_index = None

        # TF2 Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer , self)
        self.car_frame = "ego_racecar/base_link"
        self.map_frame = "map"

        # Pub/Sub gốc
        self.sub_odom = self.create_subscription(Odometry , "/ego_racecar/odom", self.odom_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.pub_text_marker = self.create_publisher(Marker, "/ten_xe", 10)
        self.pub_marker_1 = self.create_publisher(Marker, "/lookahead_marker", 10)
        self.pub_marker_2 = self.create_publisher(Marker, "/publish_vi_tri_hien_tai", 10)
        self.pub_marker_3 = self.create_publisher(MarkerArray, "/publish_duong_di", 10)

        # ==========================================
        # TÍCH HỢP LOCAL OCCUPANCY GRID
        # ==========================================
        self.pub_grid = self.create_publisher(OccupancyGrid, "/local_costmap", 10)
        self.grid_manager = OccupancyGridManager(
            x_bounds=[-0.5, 3.0], 
            y_bounds=[-2.0, 2.0], 
            cell_size=0.1, 
            obstacle_inflation_radius=0.2, 
            node_obj=self,         
            pub_obj=self.pub_grid, 
            laser_frame="ego_racecar/laser"
        )
        
        self.global_virtual_obstacles = []
        self.is_obstacle_ahead = False
        self.sub_clicked_point = self.create_subscription(
            PointStamped, 
            "/clicked_point", 
            self.clicked_point_callback, 
            10
        )
        
        self.map_timer = self.create_timer(0.1, self.update_local_map_callback)
        
        # ==========================================
        # STATE MACHINE CHUYỂN ĐỘNG & NÉ VẬT CẢN
        # ==========================================
        self.state = "NORMAL" 
        self.evasion_target_global = None 
        self.offsets_to_check = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.5, -0.5]
        
        # --- Cấu hình giảm tốc mượt mà (Velocity Profiling) ---
        self.evasion_initial_dist = 1.0  
        self.EVASION_MIN_SPEED = 1.0     # Điểm dừng tốc chậm nhất khi né (có thể tùy chỉnh)

        # Đọc Waypoint
        self.waypoints = []
        csv_path = "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        if os.path.exists(csv_path):
            self.load_waypoints(csv_path)
            self.publish_duong_di()
        else:
            self.get_logger().error(f"Khong tim thay file CSV tai: {csv_path}")

        self.get_logger().info(f"Pure Pursuit Adaptive khoi dong. T_look={self.lookahead_time}s")

    # ==========================================
    # CÁC HÀM XỬ LÝ SỰ KIỆN & BẢN ĐỒ
    # ==========================================
    def transform_local_to_map(self, local_x, local_y):
        """Chuyển tọa độ từ base_link sang map"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, 
                self.car_frame, 
                rclpy.time.Time(seconds=0),
                Duration(seconds=0.05)
            )
            p_input = PointStamped()
            p_input.header.frame_id = self.car_frame
            p_input.point.x = float(local_x)
            p_input.point.y = float(local_y)
            p_input.point.z = 0.0
            p_transformed = tf2_geometry_msgs.do_transform_point(p_input, transform)
            return np.array([p_transformed.point.x, p_transformed.point.y])
        except Exception as e:
            return None

    def clicked_point_callback(self, msg: PointStamped):
        """Nhận tọa độ từ RViz và thêm vào danh sách vật cản toàn cục"""
        x = msg.point.x
        y = msg.point.y
        self.global_virtual_obstacles.append([x, y])
        self.get_logger().info(f"📍 Da them vat can ao moi tai: Map(x={x:.2f}, y={y:.2f})")

    def update_local_map_callback(self):
        self.grid_manager.occupancy_grid.fill(0)
        local_obstacles = []
        for obs in self.global_virtual_obstacles:
            transformed = self.transform_waypoint(obs)
            if transformed is not None:
                local_obstacles.append(transformed)
                
        self.grid_manager.add_virtual_obstacles(local_obstacles)
        self.is_obstacle_ahead = self.grid_manager.check_line_collision(0.0, 0.0, 1.0, 0.0)
        self.grid_manager.publish_for_vis()

    # ==========================================
    # ĐIỀU KHIỂN CHÍNH (ODOMETRY CALLBACK)
    # ==========================================
    def odom_callback(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        current_speed = math.hypot(vx, vy)
        raw_ld = current_speed * self.lookahead_time
        self.Ld = np.clip(raw_ld, self.min_lookahead, self.max_lookahead)

        try:
            transform = self.tf_buffer.lookup_transform(self.map_frame, self.car_frame, rclpy.time.Time(), Duration(seconds=0.05))
            robot_x_map = transform.transform.translation.x
            robot_y_map = transform.transform.translation.y
            self.publish_vi_tri_hien_tai(robot_x_map, robot_y_map)
        except TransformException: return

        target_global = None

        # ------------------------------------
        # STATE MACHINE: KIỂM TRA & NÉ VẬT CẢN
        # ------------------------------------
        if self.state == "NORMAL":
            # Quét hình nón 20 độ phía trước (10 độ mỗi bên, tan(10) ~ 0.176)
            y_checks = [0.0, 0.176, -0.176] 
            obstacle_detected = False
            
            for y_scan in y_checks:
                if self.grid_manager.check_line_collision(0.0, 0.0, 1.0, y_scan):
                    obstacle_detected = True
                    break

            if obstacle_detected:
                self.get_logger().warn("Phat hien vat can! Bat dau tim duong ne...")
                found_evasion = False
                for dy in self.offsets_to_check:
                    if not self.grid_manager.check_line_collision(0.0, 0.0, 1.0, dy):
                        global_pt = self.transform_local_to_map(1.0, dy)
                        if global_pt is not None:
                            self.evasion_target_global = global_pt
                            self.state = "EVADING"
                            # Lưu lại khoảng cách ban đầu để chuẩn bị nội suy tốc độ
                            self.evasion_initial_dist = self.dist([robot_x_map, robot_y_map], global_pt) 
                            found_evasion = True
                            self.get_logger().info(f"Da tim thay diem ne tai offset Y = {dy}m")
                            break
                
                if not found_evasion:
                    self.get_logger().error("Khong tim thay duong ne! DUNG XE.")
                    stop_msg = AckermannDriveStamped()
                    self.pub_drive.publish(stop_msg)
                    return
            else:
                target_global = self.get_diem_lookahead(robot_x_map, robot_y_map)

        if self.state == "EVADING":
            target_global = self.evasion_target_global
            dist_to_evasion = self.dist([robot_x_map, robot_y_map], target_global)
            
            # Nếu đã tới điểm né (cách < 0.3m) -> Bắt đầu Recovery vào làn
            if dist_to_evasion < 0.3:
                self.get_logger().info("Da ne xong! Tim diem tiep theo de quay lai duong...")
                
                min_d = float('inf')
                closest_idx = 0
                for i, pt in enumerate(self.waypoints):
                    d = self.dist(target_global, pt)
                    if d < min_d:
                        min_d = d
                        closest_idx = i
                
                # Tiến tới trước 0.5m dọc theo đường waypoint
                accumulated_dist = 0.0
                recover_idx = closest_idx
                while accumulated_dist < 0.5:
                    next_idx = (recover_idx + 1) % len(self.waypoints)
                    accumulated_dist += self.dist(self.waypoints[recover_idx], self.waypoints[next_idx])
                    recover_idx = next_idx
                
                self.start_index = recover_idx
                self.state = "NORMAL"
                self.evasion_target_global = None
                target_global = self.get_diem_lookahead(robot_x_map, robot_y_map)

        # ------------------------------------
        # PURE PURSUIT VÀ VELOCITY PROFILING
        # ------------------------------------
        if target_global is not None:
            self.publish_lookahead_marker(target_global[0], target_global[1])
            target_local = self.transform_waypoint(target_global)

            if target_local is not None:
                x_local = target_local[0]
                y_local = target_local[1]
                Ld_square = x_local**2 + y_local**2
                
                if Ld_square < 0.001: return

                curvature = 2.0 * y_local / Ld_square
                steering_angle = math.atan(curvature * self.L) * self.kq
                steering_angle = np.clip(steering_angle, -0.35, 0.35)

                drive_msg = AckermannDriveStamped()
                drive_msg.drive.steering_angle = steering_angle
                
                # Tốc độ khi bám đường bình thường
                abs_angle = abs(steering_angle)
                normal_speed = self.slope * abs_angle + self.MAX_SPEED
                normal_speed = np.clip(normal_speed, self.MIN_SPEED, self.MAX_SPEED)
                
                if self.state == "EVADING":
                    # Tỉ lệ giảm dần từ 1.0 (xa) về 0.0 (tới đích né)
                    dist_to_evasion = self.dist([robot_x_map, robot_y_map], self.evasion_target_global)
                    ratio = dist_to_evasion / self.evasion_initial_dist
                    ratio = np.clip(ratio, 0.0, 1.0) 
                    
                    # LERP: Khi tới sát điểm né, tốc độ sẽ giảm chạm ngưỡng EVASION_MIN_SPEED
                    smooth_speed = self.EVASION_MIN_SPEED + (normal_speed - self.EVASION_MIN_SPEED) * ratio
                    drive_msg.drive.speed = smooth_speed
                else:
                    # Chạy tốc độ bình thường khi ở trạng thái NORMAL
                    drive_msg.drive.speed = normal_speed
                
                self.pub_drive.publish(drive_msg)

    # ==========================================
    # CÁC HÀM HỖ TRỢ PURE PURSUIT BẢN ĐỒ
    # ==========================================
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

    def get_diem_lookahead(self, robot_x, robot_y):
        robot_pos = np.array([robot_x, robot_y])
        min_dist = float('inf')
        if not self.waypoints: return None
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
        search_window = 40
        lookahead_point = None
        
        for i in range(search_window):
            idx_start = (nearest_idx + i) % len(self.waypoints)
            idx_end = (nearest_idx + i + 1) % len(self.waypoints)
            p1 = np.array(self.waypoints[idx_start])
            p2 = np.array(self.waypoints[idx_end])
            
            intersection = self.find_giao_diem_voi_vong_tron_ahead(p1, p2, robot_pos, self.Ld)
            if intersection is not None:
                lookahead_point = intersection
                break
                
        if lookahead_point is None:
            fallback_idx = (nearest_idx + 5) % len(self.waypoints)
            lookahead_point = np.array(self.waypoints[fallback_idx])
            
        return lookahead_point

    def transform_waypoint(self, target_point):
        if target_point is None: return None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.car_frame, 
                self.map_frame, 
                rclpy.time.Time(seconds=0),
                Duration(seconds=1.0)
            )
            p_input = PointStamped()
            p_input.header.frame_id = self.map_frame
            p_input.header.stamp = transform.header.stamp 
            p_input.point.x = float(target_point[0])
            p_input.point.y = float(target_point[1])
            p_input.point.z = 0.0
            p_transformed = tf2_geometry_msgs.do_transform_point(p_input, transform)
            return np.array([p_transformed.point.x, p_transformed.point.y])
        except (TransformException, Exception) as e:
            return None

    def dist(self, p1, p2):
        return math.sqrt((p1[0] -  p2[0])**2 + (p1[1] - p2[1])**2)

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
            self.waypoints = self.smooth_path(raw_waypoints, weight_data=0.5, weight_smooth=0.5)
            self.get_logger().info(f"Da tai {len(self.waypoints)} diem waypoint.")
        except Exception as e:
            self.get_logger().error(f"LOI DOC FILE: {e}")

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

    # ==========================================
    # VISUALIZATION HELPERS
    # ==========================================
    def publish_ten_xe(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "text_info"
        marker.id = 999
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.scale.z = 1.0
        marker.color.a = 1.0; marker.color.r = 0.0; marker.color.g = 0.5; marker.color.b = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 2.8 
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.text = "EIU FABLAB"
        self.pub_text_marker.publish(marker)

    def publish_lookahead_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3
        marker.color.a = 1.0; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        self.pub_marker_1.publish(marker)

    def publish_vi_tri_hien_tai(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "current_pos"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3
        marker.color.a = 1.0; marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        self.pub_marker_2.publish(marker)

    def publish_duong_di(self):
        marker_array = MarkerArray()
        for i, point in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.1; marker.scale.y = 0.1; marker.scale.z = 0.1
            marker.color.a = 1.0; marker.color.r = 0.5; marker.color.g = 0.5
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker_array.markers.append(marker)
        self.pub_marker_3.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    except Exception as e: print(f"Error: {e}")
    finally:
        if rclpy.ok():
            try:
                stop_msg = AckermannDriveStamped()
                node.pub_drive.publish(stop_msg)
                time.sleep(0.1)
            except: pass
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()