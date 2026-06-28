import numpy as np
import math
import random
import scipy.interpolate as interpolate

class treeNode():
    def __init__(self, x, y, is_root=False):
        self.x = x
        self.y = y
        self.is_root = is_root
        self.children = []
        self.parent = None
        self.cost = 0.0

class RRTAlgorithm():
    def __init__(self, start, goal, interations, collision_margin, steer_length, goal_tolerance, grid):
        self.start_node = treeNode(start[0], start[1], True)
        self.goal_node = treeNode(goal[0], goal[1])
        self.tree = [self.start_node]
        self.iterations = min(interations, 3000) 
        self.grid = grid
        self.margin = collision_margin
        self.steer_length = steer_length
        self.goal_tolerance = goal_tolerance
        
        # Kích thước Grid
        self.grid_h, self.grid_w = grid.shape

    def sample(self):
        # [CẢI TIẾN] Goal Bias: 10% cơ hội chọn thẳng đích
        if random.randint(0, 100) > 90:
            return np.array([self.goal_node.x, self.goal_node.y])
        
        # Lấy mẫu ngẫu nhiên (Thử 10 lần để tìm điểm trống)
        for _ in range(10): 
            rx = random.randint(0, self.grid_h - 1)
            ry = random.randint(0, self.grid_w - 1)
            if self.grid[rx, ry] == 0:
                return np.array([rx, ry])
        return None

    def nearest(self, tree, sampled_point):
        # Tìm node gần nhất theo khoảng cách bình phương (nhanh hơn sqrt)
        dists = [(node.x - sampled_point[0])**2 + (node.y - sampled_point[1])**2 for node in tree]
        return dists.index(min(dists))

    def steer(self, nearest_node, sampled_point):
        new_node = treeNode(0, 0)
        dist = math.hypot(sampled_point[0] - nearest_node.x, sampled_point[1] - nearest_node.y)
        
        if dist <= self.steer_length:
            new_node.x = sampled_point[0]
            new_node.y = sampled_point[1]
        else:
            scale = self.steer_length / dist
            new_node.x = nearest_node.x + (sampled_point[0] - nearest_node.x) * scale
            new_node.y = nearest_node.y + (sampled_point[1] - nearest_node.y) * scale
        return new_node

    def check_collision(self, n1, n2):
        x1, y1 = int(n1.x), int(n1.y)
        x2, y2 = int(n2.x), int(n2.y)
        
        # Kiểm tra các điểm trên đoạn thẳng nối n1-n2 bằng Bresenham
        points = self.get_line_points(x1, y1, x2, y2)
        
        for px, py in points:
            # Check biên
            if not (0 <= px < self.grid_h and 0 <= py < self.grid_w):
                return True
            # Check vật cản (100 là chướng ngại vật)
            if self.grid[px, py] == 100:
                return True
        return False

    def get_line_points(self, x0, y0, x1, y1):
        """Thuật toán Bresenham để lấy các điểm pixel trên đường thẳng"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def is_goal(self, node, goal_x, goal_y):
        return math.hypot(node.x - goal_x, node.y - goal_y) <= self.goal_tolerance

    def find_path_2(self, end_node):
        path = []
        curr = end_node
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        return path[::-1] # Đảo ngược từ Start -> End

    # --- HẬU XỬ LÝ (POST PROCESSING) ---
    def post_processing(self, path_nodes):
        if not path_nodes or len(path_nodes) < 2: return []
        
        # 1. Pruning (Cắt ngắn đường đi - Shortcut)
        pruned_path = [path_nodes[0]]
        curr_idx = 0
        while curr_idx < len(path_nodes) - 1:
            for i in range(len(path_nodes)-1, curr_idx, -1):
                if not self.check_collision(path_nodes[curr_idx], path_nodes[i]):
                    pruned_path.append(path_nodes[i])
                    curr_idx = i
                    break
            else:
                curr_idx += 1
                pruned_path.append(path_nodes[curr_idx])
        
        # 2. Resampling (Chia nhỏ để làm mượt tốt hơn)
        resampled_path = []
        step = 0.5
        for i in range(len(pruned_path) - 1):
            n1, n2 = pruned_path[i], pruned_path[i+1]
            dist = math.hypot(n2.x - n1.x, n2.y - n1.y)
            resampled_path.append(n1)
            if dist > step:
                num = int(dist / step)
                for j in range(1, num):
                    alpha = j / num
                    nx = n1.x + (n2.x - n1.x) * alpha
                    ny = n1.y + (n2.y - n1.y) * alpha
                    resampled_path.append(treeNode(nx, ny))
        resampled_path.append(pruned_path[-1])

        # 3. Smoothing (Làm mượt bằng B-Spline)
        coords = [[n.x, n.y] for n in resampled_path]
        return self.smoothing_path_bspline(coords)

    def smoothing_path_bspline(self, path_coords):
        if len(path_coords) < 3: return path_coords
        
        # Thêm nhiễu nhỏ để tránh lỗi duplicate points của Spline
        x = [p[0] + random.uniform(-0.01, 0.01) for p in path_coords]
        y = [p[1] + random.uniform(-0.01, 0.01) for p in path_coords]
        
        try:
            # [CẢI TIẾN] s=5.0 giúp đường đi rất mượt, phù hợp cho xe chạy nhanh
            tck, u = interpolate.splprep([x, y], k=3, s=5.0) 
            u_fine = np.linspace(0, 1, num=len(path_coords)*5)
            x_fine, y_fine = interpolate.splev(u_fine, tck)
            
            smooth_path = []
            for i in range(len(x_fine)):
                smooth_path.append([x_fine[i], y_fine[i]])
            return smooth_path
        except:
            return path_coords