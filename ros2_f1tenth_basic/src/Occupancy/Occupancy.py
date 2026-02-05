import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

class OccupancyGrid:
    def __init__(self, width_m, height_m, cell_size):
        self.cell_size = cell_size
        self.grid_w = int(width_m / cell_size)
        self.grid_h = int(height_m / cell_size)
        
        # Tạo lưới toàn số 0 (0: Trống, 1: Vật cản)
        self.grid = np.zeros((self.grid_w, self.grid_h), dtype=np.int8)
        self.inflated_grid = np.zeros_like(self.grid)
        
        print(f"--> [System] Đã khởi tạo lưới: {self.grid_w}x{self.grid_h} ô. (Mỗi ô = {cell_size}m)")

    def world_to_grid(self, x, y):
        # Chuyển đổi từ mét (m) sang chỉ số mảng (index)
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)
        
        # Kiểm tra xem điểm đó có nằm trong bản đồ không
        if 0 <= grid_x < self.grid_w and 0 <= grid_y < self.grid_h:
            return grid_x, grid_y
        return None, None

    def bresenham_line(self, x0, y0, x1, y1):
        # Thuật toán vẽ đường thẳng trên lưới (Ray Tracing)
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

    def add_line_obstacle(self, start_x, start_y, end_x, end_y):
        # 1. Đổi toạ độ mét -> lưới
        gx0, gy0 = self.world_to_grid(start_x, start_y)
        gx1, gy1 = self.world_to_grid(end_x, end_y)
        
        if gx0 is None or gx1 is None:
            print("Cảnh báo: Điểm nằm ngoài bản đồ!")
            return

        # 2. Tìm các ô nằm trên đường thẳng nối 2 điểm
        points = self.bresenham_line(gx0, gy0, gx1, gy1)
        
        # 3. Đánh dấu các ô đó là vật cản (giá trị = 1)
        for px, py in points:
            self.grid[px, py] = 1

    def inflate_obstacles(self, robot_radius_m):
        # Tính bán kính ra số ô
        radius_cells = int(robot_radius_m / self.cell_size)
        
        # Cấu trúc quét (kernel)
        struct = ndimage.generate_binary_structure(2, 2)
        
        # Thực hiện phép giãn nở (Dilation)
        self.inflated_grid = ndimage.binary_dilation(
            self.grid, 
            structure=struct, 
            iterations=radius_cells
        ).astype(np.int8)
        
        print(f"--> [System] Đã 'nở' vật cản thêm {radius_cells} ô (Bán kính robot: {robot_radius_m}m)")

    def show(self):
        print("--> [System] Đang hiển thị bản đồ...")
        plt.figure(figsize=(12, 5))
        
        # Hình 1: Bản đồ gốc
        plt.subplot(1, 2, 1)
        plt.title("Bản đồ gốc (Raw Grid)\nĐường mảnh = Tường thật")
        # .T để chuyển vị (cho đúng hướng nhìn), origin='lower' để trục y hướng lên
        plt.imshow(self.grid.T, origin='lower', cmap='Greys', interpolation='nearest')
        plt.xlabel("Trục X (ô)")
        plt.ylabel("Trục Y (ô)")
        
        # Hình 2: C-Space (Bản đồ sau khi nở rộng)
        plt.subplot(1, 2, 2)
        plt.title("Configuration Space (C-Space)\nRobot = Chất điểm, Tường = Dày lên")
        plt.imshow(self.inflated_grid.T, origin='lower', cmap='Greys', interpolation='nearest')
        plt.xlabel("Trục X (ô)")
        
        plt.tight_layout()
        plt.show() # <--- ĐÂY LÀ LỆNH QUAN TRỌNG ĐỂ CỬA SỔ HIỆN LÊN

# --- CHƯƠNG TRÌNH CHÍNH (MAIN) ---
if __name__ == "__main__":
    # 1. Tạo môi trường 10m x 10m, mỗi ô 10cm
    env = OccupancyGrid(width_m=10.0, height_m=10.0, cell_size=0.1)
    
    # 2. Xây một cái "chuồng" (Vật cản)
    # Tường dưới
    env.add_line_obstacle(2.0, 2.0, 8.0, 2.0)
    # Tường trái
    env.add_line_obstacle(2.0, 2.0, 2.0, 8.0)
    # Tường phải
    env.add_line_obstacle(8.0, 2.0, 8.0, 8.0)
    # Một chướng ngại vật chéo ở giữa
    env.add_line_obstacle(4.0, 4.0, 6.0, 6.0)
    
    # 3. Giả sử robot có bán kính 0.4m -> Nở rộng vật cản
    env.inflate_obstacles(robot_radius_m=0.4)
    
    # 4. Hiển thị kết quả
    env.show()