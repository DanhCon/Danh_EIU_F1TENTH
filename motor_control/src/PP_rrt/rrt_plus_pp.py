
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import numpy as np
import math
import csv
import os
import time

from tf2_ros import Buffer, TransformListener, TransformException
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Import thư viện RRT (Chú ý: Class tên là RRTAlgorithm)
from rrt import RRTAlgorithm, treeNode

class ContinuousLocalPlanner(Node):
    def __init__(self):
        super().__init__('continuous_local_rrt')
        
        # ================= PARAMETERS =================
        # [QUAN TRỌNG] Sửa đường dẫn này trỏ tới file CSV trên máy bạn
        self.declare_parameter("waypoint_path", "/home/danh/ros2_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv")
        
        self.declare_parameter("lookahead_dist", 2.0)       # Khoảng cách tìm goal trên map global
        self.declare_parameter("max_speed", 1.8)            # Tốc độ tối đa (m/s)
        self.declare_parameter("local_box_w", 4.0)          # Rộng 4m
        self.declare_parameter("local_box_h", 5.0)          # [CẢI TIẾN] Dài 5m để nhìn xa hơn
        
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        
        # Get params
        self.csv_path = self.get_parameter("waypoint_path").value
        self.lookahead_dist = self.get_parameter("lookahead_dist").value
        self.max_speed = self.get_parameter("max_speed").value
        self.box_w = self.get_parameter("local_box_w").value
        self.box_h = self.get_parameter("local_box_h").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        # Grid settings (Resolution 5cm)
        self.local_res = 0.05
        self.grid_w = int(self.box_w / self.local_res)
        self.grid_h = int(self.box_h / self.local_res)
        
        # Variables
        self.global_waypoints = self.load_waypoints(self.csv_path)
        self.global_map = None
        self.map_info = None
        self.scan_data = None
        self.scan_angles = None
        self.car_state = None # [x, y, yaw]
        
        self.fail_count = 0
        self.last_drive = [0.0, 0.0]

        # TF & Pub/Sub
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.viz_pub = self.create_publisher(Marker, '/local_path_viz', 10)
        self.local_map_pub = self.create_publisher(OccupancyGrid, '/local_map_debug', 10)
        
        # QoS cho Map
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy
        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Control Loop 20Hz (0.05s)
        self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info(f"Planner Started! Box Size: {self.box_w}x{self.box_h}m")

    def load_waypoints(self, path):
        if not os.path.exists(path):
            self.get_logger().error(f"FILE NOT FOUND: {path}")
            return np.array([])
        points = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2: points.append([float(row[0]), float(row[1])])
        return np.array(points)

    def map_callback(self, msg):
        self.map_info = msg.info
        w, h = msg.info.width, msg.info.height
        # Chuyển map 1D thành 2D
        self.global_map = np.array(msg.data).reshape((h, w))
        self.get_logger().info("Global Map Loaded.")

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, posinf=10.0, neginf=0.0)
        self.scan_data = ranges
        self.scan_angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

    def update_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            self.car_state = [x, y, yaw]
            return True
        except TransformException:
            return False

    def get_local_goal(self):
        cx, cy, _ = self.car_state
        dists = np.linalg.norm(self.global_waypoints - np.array([cx, cy]), axis=1)
        nearest_idx = np.argmin(dists)
        
        n_points = len(self.global_waypoints)
        for i in range(n_points):
            idx = (nearest_idx + i) % n_points
            pt = self.global_waypoints[idx]
            d = math.hypot(pt[0]-cx, pt[1]-cy)
            if d > self.lookahead_dist:
                return pt
        return self.global_waypoints[(nearest_idx + 10) % n_points]

    def build_local_grid(self, goal_local):
        grid = np.zeros((self.grid_h, self.grid_w), dtype=int)
        
        # [QUAN TRỌNG] Tọa độ xe trên Local Grid
        # Xe nằm ở giữa chiều ngang (cột), và hơi lùi về sau theo chiều dọc (hàng)
        car_gx = self.grid_h // 2
        car_gy = self.grid_w // 2
        
        # 1. TRÍCH XUẤT TỪ GLOBAL MAP (Sensor Fusion Layer 1)
        # Kỹ thuật: Chiếu từng pixel của Local Map ngược lại Global Map
        cx, cy, cyaw = self.car_state
        
        if self.global_map is not None and self.map_info is not None:
            # Tạo lưới tọa độ
            x_idxs = np.arange(self.grid_h)
            y_idxs = np.arange(self.grid_w)
            grid_x, grid_y = np.meshgrid(x_idxs, y_idxs, indexing='ij')
            
            # Chuyển Grid Local -> Mét Local
            lx = (grid_x - car_gx) * self.local_res
            ly = (grid_y - car_gy) * self.local_res
            
            # Chuyển Mét Local -> Mét Global (Xoay & Dịch)
            cos_yaw = math.cos(cyaw); sin_yaw = math.sin(cyaw)
            wx = cx + lx * cos_yaw - ly * sin_yaw
            wy = cy + lx * sin_yaw + ly * cos_yaw
            
            # Chuyển Mét Global -> Index Global Map
            g_res = self.map_info.resolution
            g_ox = self.map_info.origin.position.x
            g_oy = self.map_info.origin.position.y
            
            gx_global = ((wy - g_oy) / g_res).astype(int) # Lưu ý: Map ROS lưu data theo [y, x] hoặc [row, col]
            gy_global = ((wx - g_ox) / g_res).astype(int)
            
            # Lọc điểm hợp lệ
            valid_mask = (gx_global >= 0) & (gx_global < self.global_map.shape[0]) & \
                         (gy_global >= 0) & (gy_global < self.global_map.shape[1])
            
            # Gán giá trị: Nếu map global đen (100) -> Local cũng đen
            grid[valid_mask] = np.where(self.global_map[gx_global[valid_mask], gy_global[valid_mask]] > 50, 100, 0)

        # 2. ĐỔ DỮ LIỆU LIDAR (Sensor Fusion Layer 2)
        if self.scan_data is not None:
            # Polar -> Cartesian Local
            ox = self.scan_data * np.cos(self.scan_angles)
            oy = self.scan_data * np.sin(self.scan_angles)
            
            # Filter Box
            mask = (np.abs(ox) < self.box_h/2) & (np.abs(oy) < self.box_w/2)
            ox, oy = ox[mask], oy[mask]
            
            # Mét Local -> Grid Local
            gx = (car_gx + ox / self.local_res).astype(int)
            gy = (car_gy + oy / self.local_res).astype(int)
            
            # Inflation (25cm = 5 ô)
            inf_size = 5
            for i in range(len(gx)):
                r_min = max(0, gx[i]-inf_size); r_max = min(self.grid_h, gx[i]+inf_size)
                c_min = max(0, gy[i]-inf_size); c_max = min(self.grid_w, gy[i]+inf_size)
                grid[r_min:r_max, c_min:c_max] = 100

        # 3. CLEAR SAFETY TRIANGLE (Xóa vật cản ảo ngay mũi xe)
        for r in range(car_gx, min(self.grid_h, car_gx + 15)): 
            width = int(4 + (r - car_gx)*0.6) 
            c_left = max(0, car_gy - width)
            c_right = min(self.grid_w, car_gy + width)
            grid[r, c_left:c_right] = 0

        # 4. FUNNEL MASKING (CÁI PHỄU)
        start_width_m = 0.8
        end_width_m = float(self.box_w)
        
        for r in range(self.grid_h):
            dist_x = (r - car_gx) * self.local_res
            # Chặn phía sau đuôi xe 0.5m
            if dist_x < -0.5: 
                grid[r, :] = 100 
            else:
                progress = np.clip(dist_x / (self.box_h/2.0), 0.0, 1.0)
                current_w_m = start_width_m + progress * (end_width_m - start_width_m)
                half_width_idx = int((current_w_m / 2.0) / self.local_res)
                
                # Tô đen vùng ngoài phểu
                grid[r, 0 : max(0, car_gy - half_width_idx)] = 100
                grid[r, min(self.grid_w, car_gy + half_width_idx) : self.grid_w] = 100

        # 5. CHUYỂN GOAL
        dx = goal_local[0] - self.car_state[0]
        dy = goal_local[1] - self.car_state[1]
        
        lx = dx * math.cos(-cyaw) - dy * math.sin(-cyaw)
        ly = dx * math.sin(-cyaw) + dy * math.cos(-cyaw)
        
        goal_gx = int(car_gx + lx / self.local_res)
        goal_gy = int(car_gy + ly / self.local_res)
        goal_gx = np.clip(goal_gx, 0, self.grid_h-1)
        goal_gy = np.clip(goal_gy, 0, self.grid_w-1)
        
        # Nếu goal bị phểu cắt, kéo vào trong
        if grid[goal_gx, goal_gy] == 100:
            for offset in range(1, self.grid_w):
                left = max(0, goal_gy - offset); right = min(self.grid_w - 1, goal_gy + offset)
                if grid[goal_gx, left] == 0: goal_gy = left; break
                if grid[goal_gx, right] == 0: goal_gy = right; break

        return grid, [car_gx, car_gy], [goal_gx, goal_gy]

    def control_loop(self):
        # 1. Update Pose & Check Waypoints
        if not self.update_pose(): return
        if len(self.global_waypoints) == 0:
            self.get_logger().warn("NO WAYPOINTS LOADED! Check CSV path.", throttle_duration_sec=2.0)
            self.global_waypoints = self.load_waypoints(self.csv_path)
            return

        # 2. Get Goal & Map
        global_goal = self.get_local_goal()
        local_grid, start_idx, goal_idx = self.build_local_grid(global_goal)
        self.publish_local_map(local_grid)

        # 3. SOFT START (Tìm chỗ đứng hợp lệ)
        if local_grid[start_idx[0], start_idx[1]] == 100:
            for r in range(-5, 6):
                for c in range(-5, 6):
                    nr, nc = start_idx[0]+r, start_idx[1]+c
                    if 0 <= nr < self.grid_h and 0 <= nc < self.grid_w and local_grid[nr, nc] == 0:
                        start_idx = [nr, nc]; break
                else: continue
                break
        
        # 4. RUN RRT (STANDARD)
        rrt = RRTAlgorithm(
            start=start_idx, goal=goal_idx,
            interations=400, # Max iter
            collision_margin=0,
            steer_length=4.0, # 20cm bước nhảy
            goal_tolerance=5.0,
            grid=local_grid
        )
        
        path_grid = []
        start_time = time.time()
        
        for i in range(rrt.iterations):
            if time.time() - start_time > 0.08: break # Time limit 80ms
            
            sampled = rrt.sample()
            if sampled is None: continue
            
            nearest_idx = rrt.nearest(rrt.tree, sampled)
            nearest_node = rrt.tree[nearest_idx]
            
            new_node = rrt.steer(nearest_node, sampled)
            
            if not rrt.check_collision(nearest_node, new_node):
                new_node.parent = nearest_node # RRT thường gán trực tiếp
                rrt.tree.append(new_node)
                
                if rrt.is_goal(new_node, goal_idx[0], goal_idx[1]):
                    rrt.goal_node.parent = new_node
                    path_grid = rrt.find_path_2(rrt.goal_node)
                    break
        
        # 5. DYNAMIC PURE PURSUIT CONTROL
        if len(path_grid) > 0:
            self.fail_count = 0
            smooth_grid_path = rrt.post_processing(path_grid)
            
            # Grid -> Meter
            local_path_m = []
            for pt in smooth_grid_path:
                lx = (pt[0] - self.grid_h // 2) * self.local_res
                ly = (pt[1] - self.grid_w // 2) * self.local_res
                local_path_m.append([lx, ly])
            
            self.publish_path_viz(local_path_m)
            
            # --- DYNAMIC LOOKAHEAD & SPEED ---
            current_speed = self.last_drive[0]
            # Chạy càng nhanh nhìn càng xa
            pp_lookahead = np.clip(0.5 * abs(current_speed) + 0.6, 0.8, 2.5)
            
            target_m = local_path_m[-1]
            for pt in local_path_m:
                if math.hypot(pt[0], pt[1]) >= pp_lookahead:
                    target_m = pt; break
            
            # Tính góc lái
            curvature = 2 * target_m[1] / (math.hypot(target_m[0], target_m[1])**2 + 1e-6)
            steering_gain = 1.2 
            steering = math.atan(curvature * 0.33) * steering_gain
            steering = np.clip(steering, -0.4, 0.4)
            
            # Tính tốc độ
            speed = self.max_speed
            if abs(steering) > 0.35: speed *= 0.5   
            elif abs(steering) > 0.2: speed *= 0.7 
            
            # Low pass filter
            speed = 0.6 * self.last_drive[0] + 0.4 * speed
            
            self.last_drive = [speed, steering]
            self.publish_drive(speed, steering)
        else:
            self.fail_count += 1
            if self.fail_count < 10: # Coasting
                self.publish_drive(self.last_drive[0]*0.9, self.last_drive[1])
            else: # Reversing
                self.get_logger().warn("STUCK! REVERSING...", throttle_duration_sec=1.0)
                self.publish_drive(-0.6, -self.last_drive[1])

    def publish_drive(self, speed, angle):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(angle)
        self.drive_pub.publish(msg)

    def publish_path_viz(self, path_m):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.type = Marker.LINE_STRIP; marker.action = Marker.ADD
        marker.scale.x = 0.1; marker.color.a = 1.0; marker.color.g = 1.0
        for pt in path_m:
            p = Point(); p.x, p.y = float(pt[0]), float(pt[1]); marker.points.append(p)
        self.viz_pub.publish(marker)

    def publish_local_map(self, grid):
        msg = OccupancyGrid()
        msg.header.frame_id = self.base_frame
        msg.info.resolution = self.local_res
        msg.info.width = self.grid_w; msg.info.height = self.grid_h
        msg.info.origin.position.x = -self.box_h / 2.0
        msg.info.origin.position.y = -self.box_w / 2.0
        data = grid.flatten().astype(np.int8)
        msg.data = data.tolist()
        self.local_map_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ContinuousLocalPlanner()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()