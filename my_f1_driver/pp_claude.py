#!/usr/bin/env python3
"""
MULTI-LAYER NAVIGATION STACK
Architecture:
1. Global Layer: CSV Waypoint Following
2. Local Layer: Dynamic Window Approach (DWA) for obstacle avoidance  
3. Emergency Layer: Emergency brake

Inspired by:
- F1TENTH Lab 6 (RRT)
- TEB Local Planner
- Dynamic Window Approach Paper
"""

import numpy as np
from numpy import linalg as LA
from scipy import ndimage
import math
import csv
import os
from copy import deepcopy
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

# ROS2 Messages
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, PoseStamped, PointStamped, Pose
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray

# TF2
from tf2_ros import Buffer, TransformListener, TransformException
from tf_transformations import euler_from_quaternion

# =============================================================================
# PART 1: CORE DATA STRUCTURES
# =============================================================================

class TreeNode:
    """RRT* Node"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0
        self.is_root = False

class Trajectory:
    """Candidate trajectory for DWA"""
    def __init__(self, v, omega, cost=float('inf')):
        self.v = v          # linear velocity
        self.omega = omega  # angular velocity
        self.cost = cost
        self.path = []      # list of (x,y) points

# =============================================================================
# PART 2: OCCUPANCY GRID (Vectorized & Optimized)
# =============================================================================

class FastOccupancyGrid:
    """
    Fast occupancy grid with:
    - Vectorized operations
    - Efficient collision checking
    - Virtual obstacle support
    """
    def __init__(self, bounds_x, bounds_y, resolution, inflation_radius):
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.resolution = resolution
        self.inflation_radius = inflation_radius
        
        # Create grid
        self.width = int((bounds_x[1] - bounds_x[0]) / resolution)
        self.height = int((bounds_y[1] - bounds_y[0]) / resolution)
        self.grid = np.zeros((self.width, self.height), dtype=np.int8)
        
        # Cache for inflation kernel
        inf_cells = int(inflation_radius / resolution)
        if inf_cells > 0:
            self.inflation_kernel = self._create_circular_kernel(inf_cells)
        else:
            self.inflation_kernel = None

    def _create_circular_kernel(self, radius):
        """Create circular structuring element for inflation"""
        size = 2 * radius + 1
        kernel = np.zeros((size, size), dtype=np.int8)
        center = radius
        for i in range(size):
            for j in range(size):
                if (i - center)**2 + (j - center)**2 <= radius**2:
                    kernel[i, j] = 1
        return kernel

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        i = int((x - self.bounds_x[0]) / self.resolution)
        j = int((y - self.bounds_y[0]) / self.resolution)
        
        # Check bounds
        if 0 <= i < self.width and 0 <= j < self.height:
            return i, j
        return None, None

    def grid_to_world(self, i, j):
        """Convert grid indices to world coordinates"""
        x = self.bounds_x[0] + (i + 0.5) * self.resolution
        y = self.bounds_y[0] + (j + 0.5) * self.resolution
        return x, y

    def update_from_scan(self, scan_msg):
        """Vectorized scan processing"""
        self.grid.fill(0)
        
        ranges = np.array(scan_msg.ranges)
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        
        # Filter valid ranges
        max_range = min(scan_msg.range_max, self.bounds_x[1] * 1.2)
        valid_mask = (ranges > scan_msg.range_min) & (ranges < max_range)
        
        if not np.any(valid_mask):
            return
        
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        # Convert to Cartesian
        xs = valid_ranges * np.cos(valid_angles)
        ys = valid_ranges * np.sin(valid_angles)
        
        # Convert to grid coordinates
        is_grid = ((xs - self.bounds_x[0]) / self.resolution).astype(int)
        js_grid = ((ys - self.bounds_y[0]) / self.resolution).astype(int)
        
        # Clip to valid range
        valid = (is_grid >= 0) & (is_grid < self.width) & \
                (js_grid >= 0) & (js_grid < self.height)
        
        self.grid[is_grid[valid], js_grid[valid]] = 100
        
        # Inflate obstacles
        if self.inflation_kernel is not None:
            self.grid = ndimage.binary_dilation(
                self.grid > 0, 
                structure=self.inflation_kernel
            ).astype(np.int8) * 100

    def add_virtual_obstacles(self, obstacles_world):
        """
        Add virtual obstacles
        obstacles_world: list of (x, y) in world frame
        """
        if not obstacles_world:
            return
            
        temp_grid = np.zeros_like(self.grid)
        
        for x, y in obstacles_world:
            i, j = self.world_to_grid(x, y)
            if i is not None:
                # Draw circle
                radius_cells = int(0.3 / self.resolution)  # 30cm obstacle
                for di in range(-radius_cells, radius_cells + 1):
                    for dj in range(-radius_cells, radius_cells + 1):
                        if di**2 + dj**2 <= radius_cells**2:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.width and 0 <= nj < self.height:
                                temp_grid[ni, nj] = 100
        
        # Inflate virtual obstacles
        if self.inflation_kernel is not None:
            temp_grid = ndimage.binary_dilation(
                temp_grid > 0,
                structure=self.inflation_kernel
            ).astype(np.int8) * 100
        
        # Merge with main grid
        self.grid = np.maximum(self.grid, temp_grid)

    def is_collision(self, x, y):
        """Check if point collides"""
        i, j = self.world_to_grid(x, y)
        if i is None:
            return True
        return self.grid[i, j] > 0

    def is_line_collision(self, x1, y1, x2, y2, num_checks=10):
        """Bresenham-based line collision check"""
        i1, j1 = self.world_to_grid(x1, y1)
        i2, j2 = self.world_to_grid(x2, y2)
        
        if i1 is None or i2 is None:
            return True
        
        # Bresenham's algorithm
        points = self._bresenham_line(i1, j1, i2, j2)
        
        for i, j in points:
            if self.grid[i, j] > 0:
                return True
        return False

    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham line algorithm"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            
            if x0 == x1 and y0 == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return points

# =============================================================================
# PART 3: RRT* PLANNER (Optimized)
# =============================================================================

class RRTStarPlanner:
    """
    RRT* with:
    - Informed sampling
    - Path smoothing
    - Adaptive goal bias
    """
    def __init__(self, grid, config):
        self.grid = grid
        self.max_iter = config.get('max_iter', 500)
        self.goal_sample_rate = config.get('goal_bias', 0.1)
        self.expand_dis = config.get('expand_dis', 0.5)
        self.search_radius = config.get('search_radius', 1.0)
        self.goal_threshold = config.get('goal_threshold', 0.3)
        
    def plan(self, start, goal):
        """
        Plan path from start to goal
        Returns: list of (x,y) points or None
        """
        # Initialize tree
        start_node = TreeNode(start[0], start[1])
        start_node.is_root = True
        self.tree = [start_node]
        
        best_goal_node = None
        
        for i in range(self.max_iter):
            # Sample point
            if np.random.random() < self.goal_sample_rate:
                rnd = goal
            else:
                rnd = self._sample_free()
            
            # Find nearest node
            nearest = self._get_nearest(rnd)
            
            # Steer towards sample
            new_node = self._steer(nearest, rnd)
            
            if new_node is None:
                continue
            
            # Check collision
            if self.grid.is_line_collision(nearest.x, nearest.y, new_node.x, new_node.y):
                continue
            
            # Find best parent (RRT* optimization)
            new_node = self._choose_parent(new_node)
            
            if new_node.parent is None:
                continue
            
            # Add to tree
            self.tree.append(new_node)
            
            # Rewire tree (RRT* optimization)
            self._rewire(new_node)
            
            # Check if reached goal
            dist_to_goal = math.hypot(new_node.x - goal[0], new_node.y - goal[1])
            if dist_to_goal < self.goal_threshold:
                if best_goal_node is None or new_node.cost < best_goal_node.cost:
                    best_goal_node = new_node
        
        # Extract path
        if best_goal_node is not None:
            path = self._extract_path(best_goal_node)
            path = self._smooth_path(path)
            return path
        
        return None

    def _sample_free(self):
        """Sample random free point"""
        max_attempts = 100
        for _ in range(max_attempts):
            x = np.random.uniform(self.grid.bounds_x[0], self.grid.bounds_x[1])
            y = np.random.uniform(self.grid.bounds_y[0], self.grid.bounds_y[1])
            
            if not self.grid.is_collision(x, y):
                return (x, y)
        
        # Fallback: return center
        return (
            (self.grid.bounds_x[0] + self.grid.bounds_x[1]) / 2,
            (self.grid.bounds_y[0] + self.grid.bounds_y[1]) / 2
        )

    def _get_nearest(self, point):
        """Find nearest node in tree"""
        dists = [(node, math.hypot(node.x - point[0], node.y - point[1])) 
                 for node in self.tree]
        return min(dists, key=lambda x: x[1])[0]

    def _steer(self, from_node, to_point):
        """Steer from node towards point"""
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dist = math.hypot(dx, dy)
        
        if dist < 1e-6:
            return None
        
        # Limit expansion distance
        if dist > self.expand_dis:
            ratio = self.expand_dis / dist
            dx *= ratio
            dy *= ratio
        
        new_node = TreeNode(from_node.x + dx, from_node.y + dy)
        new_node.parent = from_node
        new_node.cost = from_node.cost + math.hypot(dx, dy)
        
        return new_node

    def _choose_parent(self, node):
        """Choose best parent from nearby nodes (RRT*)"""
        # Find nodes within search radius
        nearby = [n for n in self.tree 
                  if math.hypot(n.x - node.x, n.y - node.y) < self.search_radius]
        
        if not nearby:
            return node
        
        # Find best parent
        min_cost = node.cost
        best_parent = node.parent
        
        for near_node in nearby:
            edge_cost = math.hypot(near_node.x - node.x, near_node.y - node.y)
            new_cost = near_node.cost + edge_cost
            
            if new_cost < min_cost:
                if not self.grid.is_line_collision(near_node.x, near_node.y, node.x, node.y):
                    min_cost = new_cost
                    best_parent = near_node
        
        node.parent = best_parent
        node.cost = min_cost
        return node

    def _rewire(self, new_node):
        """Rewire tree around new node (RRT*)"""
        nearby = [n for n in self.tree 
                  if math.hypot(n.x - new_node.x, n.y - new_node.y) < self.search_radius]
        
        for near_node in nearby:
            if near_node.is_root:
                continue
            
            edge_cost = math.hypot(new_node.x - near_node.x, new_node.y - near_node.y)
            new_cost = new_node.cost + edge_cost
            
            if new_cost < near_node.cost:
                if not self.grid.is_line_collision(new_node.x, new_node.y, near_node.x, near_node.y):
                    near_node.parent = new_node
                    near_node.cost = new_cost

    def _extract_path(self, goal_node):
        """Extract path from tree"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        
        path.reverse()
        return path

    def _smooth_path(self, path, max_iter=100):
        """Smooth path using shortcut method"""
        if len(path) < 3:
            return path
        
        smoothed = path.copy()
        
        for _ in range(max_iter):
            if len(smoothed) < 3:
                break
            
            # Pick two random points
            i = np.random.randint(0, len(smoothed) - 2)
            j = np.random.randint(i + 2, len(smoothed))
            
            # Try to connect directly
            if not self.grid.is_line_collision(
                smoothed[i][0], smoothed[i][1],
                smoothed[j][0], smoothed[j][1]
            ):
                # Remove intermediate points
                smoothed = smoothed[:i+1] + smoothed[j:]
        
        return smoothed

# =============================================================================
# PART 4: DYNAMIC WINDOW APPROACH (DWA)
# =============================================================================

class DWAPlanner:
    """
    Dynamic Window Approach for local planning
    Better than pure pursuit for dense obstacles
    """
    def __init__(self, grid, config):
        self.grid = grid
        
        # Vehicle constraints
        self.max_speed = config.get('max_speed', 3.0)
        self.min_speed = config.get('min_speed', 0.0)
        self.max_accel = config.get('max_accel', 5.0)
        self.max_omega = config.get('max_omega', 1.0)  # rad/s
        self.max_alpha = config.get('max_alpha', 2.0)  # rad/s^2
        
        # DWA parameters
        self.dt = config.get('dt', 0.2)
        self.predict_time = config.get('predict_time', 2.0)
        self.v_resolution = config.get('v_reso', 0.2)
        self.omega_resolution = config.get('omega_reso', 0.1)
        
        # Cost weights
        self.w_heading = config.get('w_heading', 0.15)
        self.w_dist = config.get('w_dist', 0.5)
        self.w_velocity = config.get('w_velocity', 0.3)
        self.w_obstacle = config.get('w_obstacle', 0.05)
        
        self.wheelbase = config.get('wheelbase', 0.33)

    def compute_control(self, state, goal, current_v):
        """
        Compute optimal control
        state: (x, y, yaw)
        goal: (x, y)
        current_v: current velocity
        Returns: (v, delta) - velocity and steering angle
        """
        # Get dynamic window
        dw = self._calc_dynamic_window(current_v)
        
        # Evaluate all trajectories
        best_traj = None
        min_cost = float('inf')
        
        v_samples = np.arange(dw[0], dw[1], self.v_resolution)
        omega_samples = np.arange(dw[2], dw[3], self.omega_resolution)
        
        for v in v_samples:
            for omega in omega_samples:
                traj = self._predict_trajectory(state, v, omega)
                
                # Check collision
                if self._is_trajectory_collision(traj):
                    continue
                
                # Calculate cost
                cost = self._calc_trajectory_cost(traj, goal, v)
                
                if cost < min_cost:
                    min_cost = cost
                    best_traj = Trajectory(v, omega, cost)
                    best_traj.path = traj
        
        if best_traj is None:
            # Emergency: stop
            return 0.0, 0.0
        
        # Convert omega to steering angle (Ackermann)
        delta = math.atan2(self.wheelbase * best_traj.omega, max(best_traj.v, 0.1))
        delta = np.clip(delta, -0.4, 0.4)  # limit steering
        
        return best_traj.v, delta

    def _calc_dynamic_window(self, current_v):
        """Calculate dynamic window"""
        # Vehicle model limits
        vs_model = [self.min_speed, self.max_speed, 
                    -self.max_omega, self.max_omega]
        
        # Dynamic constraints
        vs_dynamic = [
            current_v - self.max_accel * self.dt,
            current_v + self.max_accel * self.dt,
            -self.max_omega,  # simplified
            self.max_omega
        ]
        
        # Intersection
        dw = [
            max(vs_model[0], vs_dynamic[0]),
            min(vs_model[1], vs_dynamic[1]),
            max(vs_model[2], vs_dynamic[2]),
            min(vs_model[3], vs_dynamic[3])
        ]
        
        return dw

    def _predict_trajectory(self, state, v, omega):
        """Predict trajectory"""
        x, y, yaw = state
        traj = []
        time = 0.0
        
        while time < self.predict_time:
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
            yaw += omega * self.dt
            time += self.dt
            
            traj.append((x, y))
        
        return traj

    def _is_trajectory_collision(self, traj):
        """Check trajectory collision"""
        for x, y in traj:
            if self.grid.is_collision(x, y):
                return True
        return False

    def _calc_trajectory_cost(self, traj, goal, v):
        """Calculate trajectory cost"""
        if not traj:
            return float('inf')
        
        # Heading cost
        last_x, last_y = traj[-1]
        angle_to_goal = math.atan2(goal[1] - last_y, goal[0] - last_x)
        
        # Approximate trajectory heading
        if len(traj) > 1:
            dx = last_x - traj[0][0]
            dy = last_y - traj[0][1]
            traj_heading = math.atan2(dy, dx)
        else:
            traj_heading = 0
        
        heading_cost = abs(math.atan2(
            math.sin(angle_to_goal - traj_heading),
            math.cos(angle_to_goal - traj_heading)
        ))
        
        # Distance cost (closer to goal is better)
        dist_cost = math.hypot(last_x - goal[0], last_y - goal[1])
        
        # Velocity cost (prefer higher speed)
        velocity_cost = self.max_speed - v
        
        # Obstacle cost (distance to nearest obstacle)
        obs_cost = 0
        min_dist = float('inf')
        for x, y in traj[::3]:  # sample every 3rd point
            # Check 8 surrounding cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx = x + dx * self.grid.resolution
                    ny = y + dy * self.grid.resolution
                    if self.grid.is_collision(nx, ny):
                        dist = math.hypot(dx, dy) * self.grid.resolution
                        min_dist = min(min_dist, dist)
        
        if min_dist < 0.3:  # too close
            obs_cost = 1.0 / (min_dist + 0.01)
        
        # Total cost
        cost = (self.w_heading * heading_cost + 
                self.w_dist * dist_cost + 
                self.w_velocity * velocity_cost +
                self.w_obstacle * obs_cost)
        
        return cost

# =============================================================================
# PART 5: MAIN NAVIGATION NODE
# =============================================================================

class HybridNavigator(Node):
    """
    Hybrid navigation combining:
    - Global path following
    - DWA for local avoidance
    - RRT* for complex scenarios
    """
    def __init__(self):
        super().__init__('hybrid_navigator')
        
        # Parameters
        self.declare_parameter('use_dwa', True)
        self.declare_parameter('use_rrt_fallback', True)
        self.declare_parameter('lookahead_distance', 1.5)
        self.declare_parameter('grid_resolution', 0.1)
        self.declare_parameter('grid_size', 8.0)
        
        self.use_dwa = self.get_parameter('use_dwa').value
        self.use_rrt_fallback = self.get_parameter('use_rrt_fallback').value
        self.lookahead = self.get_parameter('lookahead_distance').value
        grid_res = self.get_parameter('grid_resolution').value
        grid_size = self.get_parameter('grid_size').value
        
        # Initialize grid
        self.grid = FastOccupancyGrid(
            bounds_x=(0.0, grid_size),
            bounds_y=(-grid_size/2, grid_size/2),
            resolution=grid_res,
            inflation_radius=0.3
        )
        
        # Initialize planners
        rrt_config = {
            'max_iter': 300,
            'goal_bias': 0.15,
            'expand_dis': 0.4,
            'search_radius': 0.8,
            'goal_threshold': 0.3
        }
        self.rrt_planner = RRTStarPlanner(self.grid, rrt_config)
        
        dwa_config = {
            'max_speed': 5.0,
            'min_speed': 0.0,
            'max_accel': 8.0,
            'max_omega': 1.2,
            'dt': 0.15,
            'predict_time': 1.5,
            'v_reso': 0.3,
            'omega_reso': 0.15,
            'w_heading': 1.0,
            'w_dist': 0.8,
            'w_velocity': 0.3,
            'w_obstacle': 0.5,
            'wheelbase': 0.33
        }
        self.dwa_planner = DWAPlanner(self.grid, dwa_config)
        
        # State
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.current_pose = None
        self.current_velocity = 0.0
        self.virtual_obstacles = []
        self.local_path = None
        self.last_rrt_time = 0
        
        # Subscribers
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 1)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)
        self.create_subscription(PointStamped, '/clicked_point', self.add_obstacle_callback, 1)
        self.create_subscription(PoseStamped, '/goal_pose', self.clear_obstacles_callback, 1)
        
        # Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.path_pub = self.create_publisher(Path, '/local_path', 1)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Load waypoints
        csv_path = "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        self.load_waypoints(csv_path)
        
        self.get_logger().info("Hybrid Navigator initialized")

    def load_waypoints(self, filename):
        self.waypoints = []
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
                        self.waypoints.append([x, y])
                    except ValueError: continue
            
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")
        except Exception as e:
            self.get_logger().error(f"Error loading CSV: {e}")
            
        except FileNotFoundError:
            self.get_logger().error(f"CSV file not found: {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")

    def _smooth_waypoints(self, waypoints, weight_data=0.5, weight_smooth=0.3, tolerance=0.00001):
        """Path smoothing algorithm"""
        if len(waypoints) < 3:
            return waypoints
        
        smooth = deepcopy(waypoints)
        change = tolerance
        
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(waypoints) - 1):
                for j in range(len(waypoints[0])):
                    aux = smooth[i][j]
                    smooth[i][j] += weight_data * (waypoints[i][j] - smooth[i][j])
                    smooth[i][j] += weight_smooth * (smooth[i-1][j] + smooth[i+1][j] - 2.0 * smooth[i][j])
                    change += abs(aux - smooth[i][j])
        
        return smooth

    def scan_callback(self, msg):
        """Update occupancy grid from laser scan"""
        self.grid.update_from_scan(msg)
        
        # Add virtual obstacles if any
        if self.virtual_obstacles:
            # Transform to car frame
            obstacles_car = []
            if self.current_pose:
                for obs_map in self.virtual_obstacles:
                    obs_car = self._transform_to_car_frame(obs_map)
                    if obs_car is not None:
                        obstacles_car.append(obs_car)
            
            if obstacles_car:
                self.grid.add_virtual_obstacles(obstacles_car)
        
        # Publish grid for visualization (throttled)
        if hasattr(self, '_grid_pub_counter'):
            self._grid_pub_counter += 1
        else:
            self._grid_pub_counter = 0
        
        if self._grid_pub_counter % 20 == 0:
            self._publish_grid()

    def odom_callback(self, msg):
        """Main control loop - DEBUG VERSION"""
        # Update state
        self.current_velocity = math.hypot(
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y
        )
        
        # Get current pose
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'ego_racecar/base_link',
                rclpy.time.Time()
            )
            self.current_pose = {
                'x': t.transform.translation.x,
                'y': t.transform.translation.y,
                'yaw': euler_from_quaternion([
                    t.transform.rotation.x, t.transform.rotation.y,
                    t.transform.rotation.z, t.transform.rotation.w
                ])[2]
            }
        except TransformException:
            return
        
        if not self.waypoints:
            # self.get_logger().info("CHUA CO WAYPOINTS!") # Debug
            return
        
        # Get target
        target = self._get_lookahead_point()
        if target is None:
            # self.get_logger().info("KHONG TIM THAY TARGET!") # Debug
            return
        
        target_car = self._transform_to_car_frame(target)
        if target_car is None:
            return
        
        # === DEBUG LOGIC ===
        v_cmd = 0.0
        delta_cmd = 0.0
        
        # Check blocked
        is_blocked = self.grid.is_line_collision(0, 0, target_car[0], target_car[1])
        
        # IN RA TRẠNG THÁI ĐỂ KIỂM TRA
        # print(f"--- DEBUG ---")
        # print(f"Blocked: {is_blocked} | RRT Enabled: {self.use_rrt_fallback}")
        
        if is_blocked and self.use_rrt_fallback:
            print(">> Đang chạy Logic RRT*")
            # ... (Logic RRT giữ nguyên hoặc copy lại từ code cũ) ...
            current_time = self.get_clock().now().nanoseconds / 1e9
            if (self.local_path is None or current_time - self.last_rrt_time > 1.0):
                goal_car = self._transform_to_car_frame(target)
                if goal_car is not None:
                    path = self.rrt_planner.plan((0, 0), goal_car)
                    if path is not None:
                        self.local_path = path
                        self.last_rrt_time = current_time
                        self._publish_path(path)
            
            if self.local_path is not None:
                v_cmd, delta_cmd = self._pure_pursuit(self.local_path)

        elif self.use_dwa:
            # print(">> Đang chạy Logic DWA")
            state = (0, 0, 0)
            goal = target_car
            v_cmd, delta_cmd = self.dwa_planner.compute_control(
                state, goal, self.current_velocity
            )
            print(f"   DWA Output -> Speed: {v_cmd:.2f}, Steer: {delta_cmd:.2f}")

        else:
            print(">> Đang chạy Pure Pursuit thường")
            v_cmd, delta_cmd = self._pure_pursuit([(target_car[0], target_car[1])])
        
        # Check Phanh Khẩn Cấp (Emergency Brake)
        # Rất có thể xe bị dừng ở đây!
        if self.grid.is_line_collision(0, 0, 0.4, 0):
            print("!!! PHANH KHẨN CẤP KÍCH HOẠT !!!")
            v_cmd = 0.0
            delta_cmd = 0.0
        
        # Publish
        self._publish_drive(v_cmd, delta_cmd)

    def _get_lookahead_point(self):
        """Get lookahead waypoint"""
        if not self.waypoints or self.current_pose is None:
            return None
        
        # Find closest waypoint
        min_dist = float('inf')
        closest_idx = 0
        
        for i, wp in enumerate(self.waypoints):
            dist = math.hypot(
                wp[0] - self.current_pose['x'],
                wp[1] - self.current_pose['y']
            )
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Find lookahead point
        total_dist = 0
        for i in range(closest_idx, len(self.waypoints)):
            if i > 0:
                total_dist += math.hypot(
                    self.waypoints[i][0] - self.waypoints[i-1][0],
                    self.waypoints[i][1] - self.waypoints[i-1][1]
                )
            
            if total_dist >= self.lookahead:
                return self.waypoints[i]
        
        # Wrap around
        return self.waypoints[0]

    def _pure_pursuit(self, path):
        """Pure pursuit controller"""
        if not path:
            return 0.0, 0.0
        
        # Find lookahead point on path
        lookahead_dist = min(self.lookahead, 1.5)
        target = None
        
        for point in path:
            dist = math.hypot(point[0], point[1])
            if dist >= lookahead_dist:
                target = point
                break
        
        if target is None:
            target = path[-1]
        
        # Pure pursuit geometry
        L = 0.33  # wheelbase
        ld = math.hypot(target[0], target[1])
        
        if ld < 0.1:
            return 0.0, 0.0
        
        # Curvature
        curvature = 2.0 * target[1] / (ld * ld)
        
        # Steering angle
        delta = math.atan(curvature * L)
        delta = np.clip(delta, -0.4, 0.4)
        
        # Speed based on curvature
        if abs(delta) > 0.2:
            v = 1.5
        else:
            v = 2.5
        
        return v, delta

    def _transform_to_car_frame(self, point_map):
        """Transform point from map to car frame"""
        if self.current_pose is None:
            return None
        
        dx = point_map[0] - self.current_pose['x']
        dy = point_map[1] - self.current_pose['y']
        yaw = self.current_pose['yaw']
        
        x_car = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        y_car = dx * math.sin(-yaw) + dy * math.cos(-yaw)
        
        return (x_car, y_car)

    def _publish_drive(self, v, delta):
        """Publish drive command"""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ego_racecar/base_link'
        msg.drive.speed = float(v)
        msg.drive.steering_angle = float(delta)
        self.drive_pub.publish(msg)

    def _publish_path(self, path):
        """Publish path for visualization"""
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ego_racecar/base_link' # Hoặc 'map' tùy logic transform của bạn
        
        for point in path:
            pose = PoseStamped()
            # [FIX] Ép kiểu sang float thuần của Python để ROS không báo lỗi
            pose.pose.position.x = float(point[0]) 
            pose.pose.position.y = float(point[1])
            msg.poses.append(pose)
        
        self.path_pub.publish(msg)

    def _publish_grid(self):
        """Publish occupancy grid"""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ego_racecar/laser'
        msg.info.resolution = self.grid.resolution
        msg.info.width = self.grid.width
        msg.info.height = self.grid.height
        msg.info.origin.position.x = self.grid.bounds_x[0]
        msg.info.origin.position.y = self.grid.bounds_y[0]
        msg.data = self.grid.grid.T.flatten().tolist()
        self.grid_pub.publish(msg)

    def add_obstacle_callback(self, msg):
        """Add virtual obstacle"""
        self.virtual_obstacles.append((msg.point.x, msg.point.y))
        self.get_logger().info(f"Added obstacle at ({msg.point.x:.2f}, {msg.point.y:.2f})")

    def clear_obstacles_callback(self, msg):
        """Clear virtual obstacles"""
        self.virtual_obstacles.clear()
        self.get_logger().info("Cleared all obstacles")

# =============================================================================
# MAIN
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = HybridNavigator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
