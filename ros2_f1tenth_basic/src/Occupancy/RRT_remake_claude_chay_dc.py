#!/usr/bin/env python3
"""
ADVANCED RRT* + INTELLIGENT NUDGING + PURE PURSUIT
Version: 2.0 - PRODUCTION READY
Features:
- Persistent Nudging (3-second offset tracking)
- RRT* fallback for complex obstacles
- Emergency brake system
- Virtual obstacle support
- Full visualization
"""

import numpy as np
from numpy import linalg as LA
from scipy import ndimage
import math
import csv
import os
from copy import deepcopy

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

# ROS2 Messages
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, PoseStamped, PointStamped, Pose
from nav_msgs.msg import Odometry, OccupancyGrid
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray

# TF2
from tf2_ros import Buffer, TransformListener, TransformException
from tf_transformations import euler_from_quaternion

# Constants
DEBUG = True
INVALID_INDEX = -1

# =============================================================================
# PART 1: DATA STRUCTURES
# =============================================================================

class TreeNode:
    """RRT* Tree Node"""
    def __init__(self):
        self.x = None
        self.y = None
        self.parent = None
        self.cost = None
        self.is_root = False

# =============================================================================
# PART 2: OCCUPANCY GRID (Optimized)
# =============================================================================

class OccupancyGridManager:
    """Fast Occupancy Grid with Virtual Obstacle Support"""
    
    def __init__(self, x_bounds, y_bounds, cell_size, obstacle_inflation_radius, publisher, laser_frame):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.cell_size = cell_size
        self.publisher = publisher
        self.obstacle_inflation_radius = obstacle_inflation_radius
        self.laser_frame = laser_frame
        
        # Create grid
        num_rows = int((x_bounds[1] - x_bounds[0]) / cell_size) + 1
        num_cols = int((y_bounds[1] - y_bounds[0]) / cell_size) + 1
        self.occupancy_grid = np.zeros((num_rows, num_cols), dtype=np.int8)

    def compute_index_from_coordinates(self, x, y):
        """Convert world coordinates to grid indices"""
        i = np.int32((x - self.x_bounds[0]) / self.cell_size)
        i = self.occupancy_grid.shape[0] - i - 1  # Flip i-axis
        
        j = np.int32((y - self.y_bounds[0]) / self.cell_size)
        j = self.occupancy_grid.shape[1] - j - 1  # Flip j-axis
        
        # Bounds checking
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
        """Bresenham line collision check"""
        grid_x1, grid_y1 = self.compute_index_from_coordinates(x1, y1)
        grid_x2, grid_y2 = self.compute_index_from_coordinates(x2, y2)
        
        if grid_x1 == INVALID_INDEX or grid_y1 == INVALID_INDEX or \
           grid_x2 == INVALID_INDEX or grid_y2 == INVALID_INDEX:
            return True  # Out of bounds = collision
        
        points = self.bresenham(grid_x1, grid_y1, grid_x2, grid_y2)
        for i, j in points:
            if self.occupancy_grid[i, j] > 0:
                return True
        return False

    def bresenham(self, x1, y1, x2, y2):
        """Bresenham's line algorithm"""
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
        """Vectorized laser scan processing"""
        self.occupancy_grid.fill(0)  # Reset grid
        
        ranges = np.array(scan_msg.ranges)
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        
        # Filter valid points
        max_dist = self.x_bounds[1] * 1.5
        valid_indices = np.where((ranges < max_dist) & (ranges > 0.05))[0]
        
        if len(valid_indices) == 0:
            return
        
        valid_ranges = ranges[valid_indices]
        valid_angles = angle_min + valid_indices * angle_increment
        
        # Convert to Cartesian
        xs = valid_ranges * np.cos(valid_angles)
        ys = valid_ranges * np.sin(valid_angles)
        
        grid_rows = self.occupancy_grid.shape[0]
        grid_cols = self.occupancy_grid.shape[1]
        
        # Convert to grid indices
        i_raw = ((xs - self.x_bounds[0]) / self.cell_size).astype(int)
        j_raw = ((ys - self.y_bounds[0]) / self.cell_size).astype(int)
        
        i_indices = grid_rows - 1 - i_raw
        j_indices = grid_cols - 1 - j_raw
        
        # Clip to valid range
        mask = (i_indices >= 0) & (i_indices < grid_rows) & \
               (j_indices >= 0) & (j_indices < grid_cols)
        
        self.occupancy_grid[i_indices[mask], j_indices[mask]] = 1
        
        # Inflate obstacles
        inf_r = int(self.obstacle_inflation_radius / self.cell_size)
        if inf_r > 0:
            self.occupancy_grid = ndimage.binary_dilation(
                self.occupancy_grid, iterations=inf_r
            ).astype(np.int8)
        
        # Clear safety zone around robot
        c_i, c_j = self.compute_index_from_coordinates(0, 0)
        safe_r = int(0.3 / self.cell_size)
        if c_i != INVALID_INDEX:
            r_min = max(0, c_i - safe_r)
            r_max = min(grid_rows, c_i + safe_r)
            c_min = max(0, c_j - safe_r)
            c_max = min(grid_cols, c_j + safe_r)
            self.occupancy_grid[r_min:r_max, c_min:c_max] = 0

    def add_virtual_obstacles(self, obstacle_list_car_frame):
        """Add virtual obstacles with inflation"""
        temp_grid = np.zeros_like(self.occupancy_grid, dtype=np.int8)
        
        # Base obstacle radius (before inflation)
        radius = 0.15  # 15cm
        radius_cells = int(radius / self.cell_size)
        
        for obs_x, obs_y in obstacle_list_car_frame:
            c_i, c_j = self.compute_index_from_coordinates(obs_x, obs_y)
            
            if c_i == INVALID_INDEX:
                continue
            
            # Draw circle
            r_min = max(0, c_i - radius_cells)
            r_max = min(temp_grid.shape[0], c_i + radius_cells + 1)
            c_min = max(0, c_j - radius_cells)
            c_max = min(temp_grid.shape[1], c_j + radius_cells + 1)
            
            for i in range(r_min, r_max):
                for j in range(c_min, c_max):
                    dist = np.sqrt((i - c_i)**2 + (j - c_j)**2)
                    if dist <= radius_cells:
                        temp_grid[i, j] = 1
        
        # Inflate virtual obstacles
        inf_r = int(self.obstacle_inflation_radius / self.cell_size)
        if inf_r > 0:
            temp_grid = ndimage.binary_dilation(
                temp_grid, iterations=inf_r
            ).astype(np.int8)
        
        # Merge with main grid
        self.occupancy_grid = np.maximum(self.occupancy_grid, temp_grid)

    def publish_for_vis(self):
        """Publish grid for RViz"""
        msg = OccupancyGrid()
        msg.header.frame_id = self.laser_frame
        msg.header.stamp = self.publisher.get_clock().now().to_msg()
        msg.info.width = self.occupancy_grid.shape[0]
        msg.info.height = self.occupancy_grid.shape[1]
        msg.info.resolution = self.cell_size
        msg.info.origin.position.x = self.x_bounds[0]
        msg.info.origin.position.y = self.y_bounds[0]
        msg.info.origin.orientation.w = 1.0
        
        # Rotate for visualization
        rotated = np.rot90(self.occupancy_grid, k=1)
        flipped = np.fliplr(rotated)
        msg.data = (flipped * 100).astype(np.int8).flatten().tolist()
        
        self.publisher.publish(msg)

# =============================================================================
# PART 3: MAIN NAVIGATION NODE
# =============================================================================

class AdvancedNavigator(Node):
    """
    Advanced Navigation System with:
    - Intelligent Nudging (persistent offset)
    - RRT* for complex obstacles
    - Pure Pursuit tracking
    - Emergency brake
    """
    
    def __init__(self):
        super().__init__('advanced_navigator')
        
        # =====================================================================
        # PARAMETERS
        # =====================================================================
        
        # Vehicle parameters
        self.L = 0.38  # Wheelbase (m)
        self.kq = 1.0  # Pure pursuit gain
        
        # Speed parameters
        self.MAX_SPEED = 1.5
        self.MIN_SPEED = 1.0
        self.MAX_ANGLE = 0.35  # Max steering angle (rad)
        
        # Lookahead parameters
        self.min_lookahead = 0.4
        self.max_lookahead = 1.2
        self.lookahead_time = 0.6
        self.rrt_tracking_distance = 1.2
        
        # RRT parameters
        self.declare_parameter('lookahead_rrt', 3.0)
        self.declare_parameter('max_steer_distance', 0.5)
        self.declare_parameter('cell_size', 0.2)
        self.declare_parameter('goal_bias', 0.15)
        self.declare_parameter('goal_close_enough', 0.3)
        self.declare_parameter('obstacle_inflation_radius', 0.25)
        self.declare_parameter('num_rrt_points', 150)
        self.declare_parameter('neighborhood_radius', 0.8)
        
        # Nudging parameters
        self.declare_parameter('nudge_duration', 3.0)
        self.declare_parameter('nudge_min_shift', 0.5)
        self.declare_parameter('nudge_max_shift', 2.0)
        self.declare_parameter('nudge_speed_factor', 0.6)  # 60% of normal speed
        
        # Get parameters
        self.rrt_lookahead = self.get_parameter('lookahead_rrt').value
        self.max_steer_distance = self.get_parameter('max_steer_distance').value
        self.cell_size = self.get_parameter('cell_size').value
        self.goal_bias = self.get_parameter('goal_bias').value
        self.goal_close_enough = self.get_parameter('goal_close_enough').value
        self.obstacle_inflation_radius = self.get_parameter('obstacle_inflation_radius').value
        self.num_rrt_points = self.get_parameter('num_rrt_points').value
        self.neighborhood_radius = self.get_parameter('neighborhood_radius').value
        
        self.nudge_duration = self.get_parameter('nudge_duration').value
        self.nudge_min_shift = self.get_parameter('nudge_min_shift').value
        self.nudge_max_shift = self.get_parameter('nudge_max_shift').value
        self.nudge_speed_factor = self.get_parameter('nudge_speed_factor').value
        
        # =====================================================================
        # STATE VARIABLES
        # =====================================================================
        
        # Nudging state (NEW!)
        self.nudge_mode = False
        self.nudge_offset = 0.0
        self.nudge_start_time = 0.0
        
        # RRT state
        self.current_local_path = None
        self.tree = []
        
        # Waypoint tracking
        self.waypoints = []
        self.start_index = None
        
        # Virtual obstacles
        self.virtual_obstacles_map_frame = []
        
        # Grid state
        self.grid_formed = False
        
        # Debug counters
        self.vis_counter = 0
        self.debug_counter = 0
        
        # =====================================================================
        # TOPICS & FRAMES
        # =====================================================================
        
        self.laser_frame = "laser"
        self.map_frame = "map"
        self.car_frame = "base_link"
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 1)
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 1)
        self.obstacle_sub = self.create_subscription(
            PointStamped, '/clicked_point', self.add_obstacle_callback, 10)
        self.clear_obs_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.clear_obstacles_callback, 10)
        
        # Publishers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, "/drive", 10)
        self.grid_vis_pub = self.create_publisher(
            OccupancyGrid, '/occupancy_grid', 10)
        self.path_vis_pub = self.create_publisher(
            Marker, '/rrt/path', 10)
        self.tree_vis_pub = self.create_publisher(
            MarkerArray, '/rrt/tree', 10)
        self.obstacle_vis_pub = self.create_publisher(
            MarkerArray, '/virtual_obstacles', 10)
        self.nudge_goal_pub = self.create_publisher(
            Marker, '/nudge_goal', 10)
        self.car_marker_pub = self.create_publisher(
            Marker, '/car_marker', 10)
        self.waypoint_vis_pub = self.create_publisher(
            MarkerArray, "/global_waypoints", 10)
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # =====================================================================
        # INITIALIZE GRID
        # =====================================================================
        
        self.grid_bounds_x = (0.0, self.rrt_lookahead * 1.5)
        self.grid_bounds_y = (-self.rrt_lookahead, self.rrt_lookahead)
        
        self.occupancy_grid = OccupancyGridManager(
            self.grid_bounds_x,
            self.grid_bounds_y,
            self.cell_size,
            self.obstacle_inflation_radius,
            self.grid_vis_pub,
            self.laser_frame
        )
        
        # =====================================================================
        # LOAD WAYPOINTS
        # =====================================================================
        
        csv_path = "/home/danh/ros2_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        
        if os.path.exists(csv_path):
            self.load_waypoints(csv_path)
            self.publish_waypoint_markers()
        else:
            self.get_logger().error(f"❌ Waypoint file not found: {csv_path}")
        
        # =====================================================================
        # STARTUP MESSAGE
        # =====================================================================
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("🚗 ADVANCED NAVIGATOR v2.0 INITIALIZED")
        self.get_logger().info("=" * 70)
        self.get_logger().info("Features:")
        self.get_logger().info("  ✅ Intelligent Nudging (3-second persistence)")
        self.get_logger().info("  ✅ RRT* fallback for complex obstacles")
        self.get_logger().info("  ✅ Emergency brake system")
        self.get_logger().info("  ✅ Virtual obstacle support")
        self.get_logger().info("=" * 70)
        self.get_logger().info("Controls:")
        self.get_logger().info("  📍 Add obstacle: Click 'Publish Point' in RViz")
        self.get_logger().info("  🗑️  Clear obstacles: Click '2D Nav Goal' in RViz")
        self.get_logger().info("=" * 70)

    # =========================================================================
    # CALLBACK FUNCTIONS
    # =========================================================================

    def scan_callback(self, scan_msg: LaserScan):
        """Process laser scan and update occupancy grid"""
        # 1. Populate grid from laser
        self.occupancy_grid.populate(scan_msg)
        
        # 2. Add virtual obstacles if any
        if self.virtual_obstacles_map_frame:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.map_frame, self.car_frame, rclpy.time.Time())
                
                current_car_pose = Pose()
                current_car_pose.position.x = t.transform.translation.x
                current_car_pose.position.y = t.transform.translation.y
                current_car_pose.orientation = t.transform.rotation
                
                obs_local_list = []
                for mx, my in self.virtual_obstacles_map_frame:
                    p_local = self.transform_point_to_car_frame(
                        [mx, my], current_car_pose)
                    
                    # Only add if in grid bounds
                    if (0 < p_local[0] < self.grid_bounds_x[1]) and \
                       (self.grid_bounds_y[0] < p_local[1] < self.grid_bounds_y[1]):
                        obs_local_list.append(p_local)
                
                if obs_local_list:
                    self.occupancy_grid.add_virtual_obstacles(obs_local_list)
            
            except TransformException:
                pass
        
        self.grid_formed = True

    def odom_callback(self, pose_msg: Odometry):
        """Main control loop"""
        if not self.grid_formed or not self.waypoints:
            return
        
        # Get current pose in map frame
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.car_frame, 
                rclpy.time.Time(), Duration(seconds=0.05))
            
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            
            car_pose = Pose()
            car_pose.position.x = robot_x
            car_pose.position.y = robot_y
            car_pose.position.z = 0.0
            car_pose.orientation = transform.transform.rotation
            
            # Visualize car position
            self.visualize_car_marker(robot_x, robot_y, car_pose.orientation)
            
        except TransformException:
            return
        
        # Calculate current speed
        vx = pose_msg.twist.twist.linear.x
        vy = pose_msg.twist.twist.linear.y
        current_speed = math.hypot(vx, vy)
        
        # Calculate normal lookahead distance
        normal_Ld = np.clip(
            current_speed * self.lookahead_time,
            self.min_lookahead,
            self.max_lookahead
        )
        
        current_active_Ld = normal_Ld
        
        # Get far goal for obstacle detection
        check_dist = max(self.rrt_lookahead, 2.5)  # Extended detection range
        global_goal_map = self.get_lookahead_point(
            robot_x, robot_y, lookahead_dist=check_dist)
        
        if global_goal_map is None:
            return
        
        global_goal_car = self.transform_point_to_car_frame(
            global_goal_map, car_pose)
        
        # Clamp goal to grid bounds
        max_x_allowed = self.grid_bounds_x[1] * 0.95
        global_goal_car[0] = np.clip(global_goal_car[0], 0.0, max_x_allowed)
        
        # =====================================================================
        # PLANNING STATE MACHINE
        # =====================================================================
        
        target_point_to_drive = None
        is_avoidance_mode = False
        
        # State 1: Following RRT path
        if self.current_local_path is not None:
            if self.is_path_safe(self.current_local_path):
                is_avoidance_mode = True
                
                # Check if reached end of path
                dist_to_end = math.hypot(
                    self.current_local_path[-1].x,
                    self.current_local_path[-1].y
                )
                
                if dist_to_end < 0.4:
                    self.get_logger().info("✅ RRT path completed!")
                    self.current_local_path = None
            else:
                self.get_logger().warn("⚠️  RRT path blocked! Replanning...")
                self.current_local_path = None
        
        # State 2: Check for obstacles
        if not is_avoidance_mode:
            is_blocked = self.occupancy_grid.check_line_collision(
                0, 0, global_goal_car[0], global_goal_car[1])
            
            if is_blocked:
                # Path blocked → Try nudging first
                nudged_map = self.try_nudging_goal(global_goal_car, car_pose)
                
                if nudged_map is not None:
                    # Nudging successful
                    target_point_to_drive = nudged_map
                    
                    # Reduce speed when nudging
                    if self.nudge_mode:
                        self.MAX_SPEED = 1.5 * self.nudge_speed_factor
                        self.MIN_SPEED = 1.2 * self.nudge_speed_factor
                else:
                    # Nudging failed → Use RRT*
                    self.get_logger().warn("❌ NUDGE FAILED! Activating RRT*")
                    
                    # Reset nudge state
                    self.nudge_mode = False
                    self.nudge_offset = 0.0
                    
                    # Plan with RRT*
                    path_nodes = self.rrt_star(global_goal_car)
                    
                    if path_nodes and len(path_nodes) > 1:
                        self.current_local_path = path_nodes
                        is_avoidance_mode = True
                        
                        if DEBUG:
                            self.visualize_path(path_nodes)
                            self.visualize_tree()
            else:
                # Path clear → Exit nudge mode if active
                if self.nudge_mode:
                    current_time = self.get_clock().now().nanoseconds / 1e9
                    elapsed = current_time - self.nudge_start_time
                    
                    if elapsed >= self.nudge_duration:
                        self.get_logger().info("✅ Nudge completed!")
                        self.nudge_mode = False
                        self.nudge_offset = 0.0
                
                # Restore normal speed
                self.MAX_SPEED = 1.5
                self.MIN_SPEED = 1.0
        
        # State 3: Select target point
        if is_avoidance_mode and self.current_local_path:
            # Follow RRT path
            current_active_Ld = self.rrt_tracking_distance
            best_node = self.find_waypoint_to_track_rrt(
                self.current_local_path, current_active_Ld)
            target_point_to_drive = self.transform_point_to_map_frame(
                [best_node.x, best_node.y], car_pose)
        
        elif target_point_to_drive is None:
            # Follow global waypoints
            short_goal_map = self.get_lookahead_point(
                robot_x, robot_y, lookahead_dist=normal_Ld)
            
            if short_goal_map is not None:
                target_point_to_drive = short_goal_map
            current_active_Ld = normal_Ld
        
        # Update lookahead for controller
        self.Ld = current_active_Ld
        
        # Execute control
        if target_point_to_drive is not None:
            # Visualize target
            target_car = self.transform_point_to_car_frame(
                target_point_to_drive, car_pose)
            self.visualize_nudge_goal(target_car)
            
            # Drive
            self.drive_pure_pursuit(
                target_point_to_drive, robot_x, robot_y, car_pose)
        
        # Debug info
        self.print_debug_info(current_speed, robot_x, robot_y)

    def add_obstacle_callback(self, msg):
        """Add virtual obstacle at clicked point"""
        x, y = msg.point.x, msg.point.y
        self.virtual_obstacles_map_frame.append((x, y))
        
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"✅ OBSTACLE ADDED #{len(self.virtual_obstacles_map_frame)}")
        self.get_logger().info(f"📍 Position: ({x:.2f}, {y:.2f})")
        self.get_logger().info("=" * 60)
        
        self.visualize_obstacles()

    def clear_obstacles_callback(self, msg):
        """Clear all virtual obstacles"""
        count = len(self.virtual_obstacles_map_frame)
        self.virtual_obstacles_map_frame.clear()
        self.current_local_path = None
        self.nudge_mode = False
        self.nudge_offset = 0.0
        
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"🗑️  CLEARED {count} OBSTACLES")
        self.get_logger().info("=" * 60)
        
        self.visualize_obstacles()

    # =========================================================================
    # NUDGING LOGIC (NEW & IMPROVED!)
    # =========================================================================

    def try_nudging_goal(self, original_goal_car, car_pose):
        """
        Intelligent nudging with persistent offset
        Returns: nudged goal in map frame or None
        """
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # If already nudging and within duration → Keep current offset
        if self.nudge_mode:
            time_elapsed = current_time - self.nudge_start_time
            
            if time_elapsed < self.nudge_duration:
                # Continue nudging with same offset
                candidate_x = original_goal_car[0]
                candidate_y = original_goal_car[1] + self.nudge_offset
                
                # Check if path still safe
                if not self.occupancy_grid.check_line_collision(
                    0, 0, candidate_x, candidate_y):
                    return self.transform_point_to_map_frame(
                        np.array([candidate_x, candidate_y]), car_pose)
                else:
                    # Path blocked → Abort nudge
                    self.get_logger().warn("⚠️  Nudge path blocked!")
                    self.nudge_mode = False
                    return None
            else:
                # Duration expired → Exit nudge mode
                self.get_logger().info("✅ Nudge duration completed")
                self.nudge_mode = False
                self.nudge_offset = 0.0
        
        # Not nudging → Try to find new offset
        if not self.nudge_mode:
            # Generate shift candidates (small to large)
            shifts = self._generate_shift_candidates()
            
            for shift in shifts:
                candidate_x = original_goal_car[0]
                candidate_y = original_goal_car[1] + shift
                
                # Check full path to goal
                if not self.occupancy_grid.check_line_collision(
                    0, 0, candidate_x, candidate_y):
                    
                    # Found safe path
                    if abs(shift) > 0.1:  # Actual nudge needed
                        self.nudge_mode = True
                        self.nudge_offset = shift
                        self.nudge_start_time = current_time
                        
                        self.get_logger().info("=" * 60)
                        self.get_logger().info(f"🔄 NUDGING ACTIVATED")
                        self.get_logger().info(f"   Offset: {shift:.2f}m")
                        self.get_logger().info(f"   Duration: {self.nudge_duration:.1f}s")
                        self.get_logger().info("=" * 60)
                    
                    return self.transform_point_to_map_frame(
                        np.array([candidate_x, candidate_y]), car_pose)
        
        # No safe nudge path found
        return None

    def _generate_shift_candidates(self):
        """Generate nudge offset candidates from small to large"""
        shifts = [0.0]  # Try straight first
        
        # Generate incremental shifts
        num_steps = 8
        for i in range(1, num_steps + 1):
            offset = self.nudge_min_shift + \
                     (self.nudge_max_shift - self.nudge_min_shift) * i / num_steps
            shifts.extend([-offset, offset])  # Try both sides
        
        return shifts

    # =========================================================================
    # RRT* PLANNER
    # =========================================================================

    def rrt_star(self, goal):
        """RRT* planning algorithm"""
        # Initialize tree
        start_node = TreeNode()
        start_node.x, start_node.y = 0.0, 0.0
        start_node.cost = 0
        start_node.is_root = True
        self.tree = [start_node]
        
        goal_with_min_cost = None
        
        # Build tree
        for _ in range(self.num_rrt_points):
            # Sample point
            sampled_point = self.sample(goal)
            
            # Find nearest node
            nearest_node = self.nearest(self.tree, sampled_point)
            
            # Steer towards sample
            new_node = self.steer(nearest_node, sampled_point)
            
            if new_node is None:
                continue
            
            # Check collision
            if self.check_collision(nearest_node, new_node):
                continue
            
            # Choose best parent (RRT* optimization)
            new_node.cost, new_node.parent = self.calc_cost_new_node(
                self.tree, new_node)
            
            if new_node.parent is None:
                continue
            
            # Add to tree
            self.tree.append(new_node)
            
            # Rewire tree (RRT* optimization)
            self.rewire(self.tree, new_node)
            
            # Check if reached goal
            if self.is_goal(new_node, goal[0], goal[1]):
                if goal_with_min_cost is None or \
                   new_node.cost < goal_with_min_cost.cost:
                    goal_with_min_cost = new_node
        
        # Extract path
        if goal_with_min_cost:
            return self.find_path(self.tree, goal_with_min_cost)
        
        return None

    def sample(self, goal):
        """Sample random point biased towards goal"""
        if np.random.uniform() < self.goal_bias:
            return (goal[0], goal[1])
        
        # Gaussian sampling around midpoint
        mean = np.array([goal[0], goal[1]]) / 2
        std_dev = LA.norm(np.array([goal[0], goal[1]])) / 2
        
        x = np.random.normal(mean[0], std_dev)
        y = np.random.normal(mean[1], std_dev / 2)
        
        x = np.clip(x, 0.0, self.grid_bounds_x[1])
        y = np.clip(y, self.grid_bounds_y[0], self.grid_bounds_y[1])
        
        return (x, y)

    def nearest(self, tree, sampled_point):
        """Find nearest node in tree"""
        min_dist = float('inf')
        nearest_node = None
        
        for node in tree:
            dist = LA.norm(
                np.array([node.x, node.y]) - np.array(sampled_point))
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        return nearest_node

    def steer(self, from_node, to_point):
        """Steer from node towards point"""
        vec = np.array(to_point) - np.array([from_node.x, from_node.y])
        dist = LA.norm(vec)
        
        if dist < 1e-6:
            return None
        
        vec = vec / dist
        
        new_node = TreeNode()
        if dist > self.max_steer_distance:
            new_node.x = from_node.x + self.max_steer_distance * vec[0]
            new_node.y = from_node.y + self.max_steer_distance * vec[1]
        else:
            new_node.x = to_point[0]
            new_node.y = to_point[1]
        
        new_node.parent = from_node
        new_node.cost = from_node.cost + LA.norm(vec * min(dist, self.max_steer_distance))
        
        return new_node

    def is_goal(self, node, gx, gy):
        """Check if node reached goal"""
        dist = LA.norm(np.array([node.x, node.y]) - np.array([gx, gy]))
        return dist <= self.goal_close_enough

    def find_path(self, tree, goal_node):
        """Extract path from tree"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current)
            current = current.parent
        
        path.reverse()
        return path

    def calc_cost_new_node(self, tree, node):
        """Calculate cost for new node (RRT*)"""
        if node.is_root:
            return 0, None
        
        neighborhood = self.near(tree, node)
        min_cost = float('inf')
        best_parent = None
        
        for n in neighborhood:
            cost = n.cost + self.line_cost(n, node)
            if cost < min_cost:
                if not self.check_collision(n, node):
                    min_cost = cost
                    best_parent = n
        
        return min_cost, best_parent

    def rewire(self, tree, node):
        """Rewire tree around new node (RRT*)"""
        neighborhood = self.near(tree, node)
        
        for n in neighborhood:
            if n.is_root:
                continue
            
            new_cost = node.cost + self.line_cost(node, n)
            if new_cost < n.cost:
                if not self.check_collision(node, n):
                    n.parent = node
                    n.cost = new_cost

    def near(self, tree, node):
        """Find nearby nodes"""
        neighborhood = []
        for n in tree:
            dist = LA.norm(
                np.array([n.x, n.y]) - np.array([node.x, node.y]))
            if dist <= self.neighborhood_radius:
                neighborhood.append(n)
        return neighborhood

    def line_cost(self, n1, n2):
        """Calculate line cost between nodes"""
        return LA.norm(
            np.array([n1.x, n1.y]) - np.array([n2.x, n2.y]))

    def check_collision(self, n1, n2):
        """Check collision between two nodes"""
        return self.occupancy_grid.check_line_collision(
            n1.x, n1.y, n2.x, n2.y)

    def is_path_safe(self, path):
        """Check if RRT path is still collision-free"""
        if path is None or len(path) < 2:
            return False
        
        for i in range(len(path) - 1):
            if self.check_collision(path[i], path[i + 1]):
                return False
        
        return True

    def find_waypoint_to_track_rrt(self, path, Ld):
        """Find waypoint on RRT path at lookahead distance"""
        best_node = path[-1]
        
        for node in path:
            dist = math.hypot(node.x, node.y)
            if dist >= Ld:
                best_node = node
                break
        
        return best_node

    # =========================================================================
    # PURE PURSUIT CONTROLLER
    # =========================================================================

    def drive_pure_pursuit(self, target_map, robot_x, robot_y, car_pose_msg):
        """Pure pursuit controller with emergency brake"""
        # Emergency brake check
        if self.occupancy_grid.check_line_collision(0.1, 0.0, 0.4, 0.0):
            self.get_logger().warn("🚨 EMERGENCY BRAKE!")
            self.publish_drive_command(-1.0, 0.0)
            return
        
        # Transform target to car frame
        target_local = self.transform_point_to_car_frame(target_map, car_pose_msg)
        x_local, y_local = target_local[0], target_local[1]
        actual_dist = math.hypot(x_local, y_local)
        
        if actual_dist < 0.01:
            return
        
        # Rescale to lookahead distance
        effective_Ld = self.Ld
        if actual_dist > effective_Ld:
            scale = effective_Ld / actual_dist
            x_local *= scale
            y_local *= scale
            Ld_square = effective_Ld ** 2
        else:
            Ld_square = actual_dist ** 2
        
        # Calculate curvature
        curvature = 2.0 * y_local / Ld_square
        
        # Calculate steering angle
        steering_angle = math.atan(curvature * self.L) * self.kq
        steering_angle = np.clip(steering_angle, -self.MAX_ANGLE, self.MAX_ANGLE)
        
        # Speed based on curvature
        if abs(steering_angle) > 0.2:
            speed = self.MIN_SPEED
        else:
            speed = self.MAX_SPEED
        
        # Publish command
        self.publish_drive_command(speed, steering_angle)

    def publish_drive_command(self, speed, steering_angle):
        """Publish Ackermann drive command"""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.car_frame
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

    # =========================================================================
    # WAYPOINT MANAGEMENT
    # =========================================================================

    def load_waypoints(self, filename):
        """Load waypoints from CSV"""
        raw_waypoints = []
        
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                
                for row in reader:
                    if not row:
                        continue
                    
                    # Handle different CSV formats
                    if len(row) == 1 and isinstance(row[0], str):
                        line_data = row[0].split()
                    else:
                        line_data = row
                    
                    try:
                        x = float(line_data[0])
                        y = float(line_data[1])
                        raw_waypoints.append([x, y])
                    except (ValueError, IndexError):
                        continue
            
            # Smooth waypoints
            self.waypoints = self.smooth_path(raw_waypoints)
            
            self.get_logger().info(f"✅ Loaded {len(self.waypoints)} waypoints")
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load waypoints: {e}")

    def smooth_path(self, path, weight_data=0.5, weight_smooth=0.2, tolerance=0.00001):
        """Smooth path using gradient descent"""
        if len(path) < 3:
            return path
        
        new_path = deepcopy(path)
        change = tolerance
        
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path) - 1):
                for j in range(len(path[0])):
                    aux = new_path[i][j]
                    new_path[i][j] += weight_data * (path[i][j] - new_path[i][j])
                    new_path[i][j] += weight_smooth * (
                        new_path[i-1][j] + new_path[i+1][j] - 2.0 * new_path[i][j])
                    change += abs(aux - new_path[i][j])
        
        return new_path

    def get_lookahead_point(self, robot_x, robot_y, lookahead_dist):
        """Get lookahead point on global path"""
        robot_pos = np.array([robot_x, robot_y])
        
        # Find closest waypoint
        if self.start_index is None:
            min_dist = float('inf')
            for i, point in enumerate(self.waypoints):
                d = self.dist([robot_x, robot_y], point)
                if d < min_dist:
                    min_dist = d
                    self.start_index = i
        else:
            # Update closest waypoint
            curr_dist = self.dist([robot_x, robot_y], self.waypoints[self.start_index])
            for _ in range(40):
                next_idx = (self.start_index + 1) % len(self.waypoints)
                next_dist = self.dist([robot_x, robot_y], self.waypoints[next_idx])
                if next_dist < curr_dist:
                    self.start_index = next_idx
                    curr_dist = next_dist
                else:
                    break
        
        # Find lookahead point
        nearest_idx = self.start_index
        lookahead_point = None
        
        for i in range(40):
            idx_start = (nearest_idx + i) % len(self.waypoints)
            idx_end = (nearest_idx + i + 1) % len(self.waypoints)
            
            p1 = np.array(self.waypoints[idx_start])
            p2 = np.array(self.waypoints[idx_end])
            
            intersection = self.find_circle_intersection(
                p1, p2, robot_pos, lookahead_dist)
            
            if intersection is not None:
                lookahead_point = intersection
                break
        
        # Fallback
        if lookahead_point is None:
            fallback_idx = (nearest_idx + 5) % len(self.waypoints)
            lookahead_point = np.array(self.waypoints[fallback_idx])
        
        return lookahead_point

    def find_circle_intersection(self, p1, p2, center, radius):
        """Find intersection of line segment with circle"""
        d = p2 - p1
        f = p1 - center
        
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None
        
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        if 0 <= t2 <= 1:
            return p1 + t2 * d
        elif 0 <= t1 <= 1:
            return p1 + t1 * d
        
        return None

    def dist(self, p1, p2):
        """Euclidean distance"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # =========================================================================
    # COORDINATE TRANSFORMATIONS
    # =========================================================================

    def transform_point_to_car_frame(self, point, car_pose_msg):
        """Transform point from map frame to car frame"""
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
        """Transform point from car frame to map frame"""
        cx = car_pose_msg.position.x
        cy = car_pose_msg.position.y
        
        q = [car_pose_msg.orientation.x, car_pose_msg.orientation.y,
             car_pose_msg.orientation.z, car_pose_msg.orientation.w]
        yaw = euler_from_quaternion(q)[2]
        
        x_map = point[0] * np.cos(yaw) - point[1] * np.sin(yaw) + cx
        y_map = point[0] * np.sin(yaw) + point[1] * np.cos(yaw) + cy
        
        return np.array([x_map, y_map])

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def visualize_obstacles(self):
        """Visualize virtual obstacles as red cylinders"""
        marker_array = MarkerArray()
        
        # Delete old markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Add new markers
        for i, (x, y) in enumerate(self.virtual_obstacles_map_frame):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.2
            
            marker.scale.x = 0.6
            marker.scale.y = 0.6
            marker.scale.z = 0.4
            
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
        
        self.obstacle_vis_pub.publish(marker_array)

    def visualize_nudge_goal(self, goal_car):
        """Visualize nudge target point"""
        marker = Marker()
        marker.header.frame_id = self.car_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        if self.nudge_mode:
            # Orange when nudging
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
        else:
            # Cyan when normal
            marker.color.r = 0.0
            marker.color.g = 0.8
            marker.color.b = 1.0
        
        marker.color.a = 1.0
        marker.pose.position.x = goal_car[0]
        marker.pose.position.y = goal_car[1]
        
        self.nudge_goal_pub.publish(marker)

    def visualize_car_marker(self, x, y, orientation):
        """Visualize car as green arrow"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "car"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.pose.orientation = orientation
        
        marker.scale.x = 0.5
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.car_marker_pub.publish(marker)

    def visualize_path(self, path):
        """Visualize RRT path as yellow line"""
        marker = Marker()
        marker.header.frame_id = self.laser_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = 0.08
        
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        for node in path:
            p = Point()
            p.x, p.y, p.z = node.x, node.y, 0.0
            marker.points.append(p)
        
        self.path_vis_pub.publish(marker)

    def visualize_tree(self):
        """Visualize RRT tree as blue points/lines"""
        marker_array = MarkerArray()
        
        # Points
        points_marker = Marker()
        points_marker.header.frame_id = self.laser_frame
        points_marker.header.stamp = self.get_clock().now().to_msg()
        points_marker.ns = "rrt_nodes"
        points_marker.id = 0
        points_marker.type = Marker.POINTS
        points_marker.action = Marker.ADD
        points_marker.scale.x = 0.05
        points_marker.scale.y = 0.05
        points_marker.color.r = 0.0
        points_marker.color.g = 0.5
        points_marker.color.b = 1.0
        points_marker.color.a = 0.6
        
        # Lines
        lines_marker = Marker()
        lines_marker.header = points_marker.header
        lines_marker.ns = "rrt_edges"
        lines_marker.id = 1
        lines_marker.type = Marker.LINE_LIST
        lines_marker.action = Marker.ADD
        lines_marker.scale.x = 0.02
        lines_marker.color = points_marker.color
        
        for node in self.tree:
            p = Point()
            p.x, p.y, p.z = node.x, node.y, 0.0
            points_marker.points.append(p)
            
            if node.parent is not None:
                p1 = Point()
                p1.x, p1.y, p1.z = node.parent.x, node.parent.y, 0.0
                p2 = Point()
                p2.x, p2.y, p2.z = node.x, node.y, 0.0
                lines_marker.points.extend([p1, p2])
        
        marker_array.markers.extend([points_marker, lines_marker])
        self.tree_vis_pub.publish(marker_array)

    def publish_waypoint_markers(self):
        """Publish global waypoints as green spheres"""
        marker_array = MarkerArray()
        
        for i, point in enumerate(self.waypoints):
            if i % 3 == 0:  # Subsample for visualization
                marker = Marker()
                marker.header.frame_id = "map"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.4
                
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1]
                
                marker_array.markers.append(marker)
        
        self.waypoint_vis_pub.publish(marker_array)

    def print_debug_info(self, speed, x, y):
        """Print debug information"""
        self.debug_counter += 1
        
        if self.debug_counter % 30 == 0:
            self.get_logger().info("=" * 70)
            self.get_logger().info(f"🚗 Speed: {speed:.2f} m/s")
            self.get_logger().info(f"📍 Position: ({x:.2f}, {y:.2f})")
            
            if self.nudge_mode:
                elapsed = (self.get_clock().now().nanoseconds / 1e9) - \
                          self.nudge_start_time
                self.get_logger().info(
                    f"🔄 NUDGING: offset={self.nudge_offset:.2f}m, "
                    f"time={elapsed:.1f}/{self.nudge_duration:.1f}s")
            
            if self.current_local_path:
                self.get_logger().info(
                    f"🛤️  RRT: {len(self.current_local_path)} waypoints")
            
            self.get_logger().info(
                f"🚧 Virtual Obstacles: {len(self.virtual_obstacles_map_frame)}")
            
            # Grid occupancy
            occupied = np.sum(self.occupancy_grid.occupancy_grid > 0)
            total = self.occupancy_grid.occupancy_grid.size
            self.get_logger().info(
                f"🗺️  Grid: {occupied}/{total} cells ({100*occupied/total:.1f}%)")
            self.get_logger().info("=" * 70)

# =============================================================================
# MAIN
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedNavigator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()