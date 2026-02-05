"""
RRT* Path Planning Simulation (Animation Version)
Author: Gemini & User
Description: RRT* có hiển thị quá trình chạy thời gian thực.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PHẦN 1: BẢN ĐỒ (Occupancy Grid)
# ==========================================
class OccupancyGrid:
    def __init__(self, width_m, height_m, cell_size):
        self.cell_size = cell_size
        self.grid_w = int(width_m / cell_size)
        self.grid_h = int(height_m / cell_size)
        self.x_bounds = (0, width_m)
        self.y_bounds = (0, height_m)
        self.grid = np.zeros((self.grid_w, self.grid_h), dtype=np.int8)

    def world_to_grid(self, x, y):
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)
        if 0 <= grid_x < self.grid_w and 0 <= grid_y < self.grid_h:
            return grid_x, grid_y
        return None, None

    def bresenham_line(self, x0, y0, x1, y1):
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

    def add_rect_obstacle(self, x, y, w, h):
        gx, gy = self.world_to_grid(x, y)
        gw = int(w / self.cell_size)
        gh = int(h / self.cell_size)
        if gx is not None and gy is not None:
            end_x = min(gx + gw, self.grid_w)
            end_y = min(gy + gh, self.grid_h)
            self.grid[gx:end_x, gy:end_y] = 1

    def check_line_collision(self, x1, y1, x2, y2):
        print("điid")
        gx1, gy1 = self.world_to_grid(x1, y1)
        gx2, gy2 = self.world_to_grid(x2, y2)
        if gx1 is None or gx2 is None: return True 
        points = self.bresenham_line(gx1, gy1, gx2, gy2)
        for px, py in points:
            if self.grid[px, py] == 1:
                return True 
        return False

# ==========================================
# PHẦN 2: NODE
# ==========================================
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

# ==========================================
# PHẦN 3: RRT* PLANNER (Có Animation)
# ==========================================
class RRTStarPlanner:
    def __init__(self, start, goal, occupancy_grid, step_size=1.0, max_iter=500, search_radius=5.0):
        self.start = Node(start[0], start[1])
        self.goal = goal 
        self.map = occupancy_grid
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius 
        self.node_list = [] 

    def get_distance(self, node1, node2):
        return math.hypot(node1.x - node2.x, node1.y - node2.y)

    def sample_point(self):
        if np.random.random() < 0.1: return self.goal
        x = np.random.uniform(self.map.x_bounds[0], self.map.x_bounds[1])
        y = np.random.uniform(self.map.y_bounds[0], self.map.y_bounds[1])
        return (x, y)

    def get_nearest_node_index(self, point):
        dlist = [(node.x - point[0])**2 + (node.y - point[1])**2 for node in self.node_list]
        return dlist.index(min(dlist))

    def steer(self, from_node, to_point):
        new_node = Node(from_node.x, from_node.y)
        d = math.hypot(to_point[0] - from_node.x, to_point[1] - from_node.y)
        if d < 1e-3: return from_node
        new_node.x += self.step_size * (to_point[0] - from_node.x) / d
        new_node.y += self.step_size * (to_point[1] - from_node.y) / d
        new_node.parent = from_node
        new_node.cost = from_node.cost + self.step_size
        return new_node

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = 50.0 * math.sqrt((math.log(nnode) / nnode))
        r = min(r, self.search_radius)
        near_inds = []
        for i, node in enumerate(self.node_list):
            if self.get_distance(node, new_node) <= r:
                near_inds.append(i)
        return near_inds

    def choose_parent(self, new_node, near_inds):
        if not near_inds: return new_node
        for i in near_inds:
            near_node = self.node_list[i]
            t_cost = near_node.cost + self.get_distance(near_node, new_node)
            if t_cost < new_node.cost:
                if not self.map.check_line_collision(near_node.x, near_node.y, new_node.x, new_node.y):
                    new_node.cost = t_cost
                    new_node.parent = near_node
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            new_cost = new_node.cost + self.get_distance(new_node, near_node)
            if new_cost < near_node.cost:
                if not self.map.check_line_collision(new_node.x, new_node.y, near_node.x, near_node.y):
                    near_node.parent = new_node
                    near_node.cost = new_cost
                    # Lưu ý: Khi vẽ animation, việc vẽ lại dây rewire rất phức tạp nên ta sẽ chỉ vẽ các node mới

    def generate_final_course(self, goal_ind):
        path = [[self.goal[0], self.goal[1]]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def planning_with_animation(self):
        """Hàm Planning có tích hợp vẽ hình"""
        self.node_list = [self.start]
        
        # --- SETUP ĐỒ HOẠ ---
        plt.ion() # Bật chế độ Interactive
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Vẽ map nền
        ax.imshow(self.map.grid.T, origin='lower', cmap='Greys', extent=[0, 20, 0, 20])
        ax.plot(self.start.x, self.start.y, "bo", markersize=8, label="Start")
        ax.plot(self.goal[0], self.goal[1], "yo", markersize=8, label="Goal")
        ax.set_title("RRT* Animation (Đang chạy...)")
        ax.grid(True)
        plt.draw()
        
        print("RRT* bắt đầu chạy (Quan sát cửa sổ đồ hoạ)...")

        for i in range(self.max_iter):
            rnd = self.sample_point()
            nearest_ind = self.get_nearest_node_index(rnd)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd)
            
            if not self.map.check_line_collision(nearest_node.x, nearest_node.y, new_node.x, new_node.y):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                self.node_list.append(new_node)
                self.rewire(new_node, near_inds)
                
                # --- VẼ MỖI BƯỚC (ANIMATION) ---
                # Chỉ vẽ khi node được thêm vào thành công
                if i % 2 == 0: # Vẽ mỗi 2 bước để không quá lag
                    if new_node.parent:
                        ax.plot([new_node.x, new_node.parent.x], 
                                [new_node.y, new_node.parent.y], "-g", alpha=0.5, linewidth=0.5)
                        plt.pause(0.001) # Tạm dừng cực ngắn để update màn hình

            # Kiểm tra về đích
            if i % 10 == 0:
                last_node = self.node_list[-1]
                dist_to_goal = math.hypot(last_node.x - self.goal[0], last_node.y - self.goal[1])
                if dist_to_goal <= self.step_size:
                    print(f"--> Tìm thấy đích ở bước {i}!")
                    final_node = self.choose_parent(Node(self.goal[0], self.goal[1]), [len(self.node_list)-1])
                    path = self.generate_final_course(len(self.node_list)-1)
                    
                    # Vẽ đường cuối cùng
                    path = np.array(path)
                    ax.plot(path[:, 0], path[:, 1], "-r", linewidth=3, label="Final Path")
                    ax.set_title(f"RRT* DONE! Steps: {i}")
                    plt.ioff() # Tắt chế độ interactive
                    plt.show() # Giữ cửa sổ lại
                    return path

        print("Hết vòng lặp, không tìm thấy đường.")
        plt.ioff()
        plt.show()
        return None

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    env = OccupancyGrid(20, 20, 0.5)
    
    # Tạo map chữ U
    env.add_rect_obstacle(x=5, y=5, w=10, h=2)
    env.add_rect_obstacle(x=13, y=5, w=2, h=8)
    env.add_rect_obstacle(x=5, y=5, w=2, h=8)
    
    # Chạy Planner với Animation
    planner = RRTStarPlanner(
        start=(2.5, 2.5), 
        goal=(18, 18), 
        occupancy_grid=env, 
        step_size=0.8, # Giảm step size chút cho cây mọc dày dễ nhìn
        max_iter=1000
    )
    
    path = planner.planning_with_animation()