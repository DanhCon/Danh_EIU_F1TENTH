#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
import csv
import os
from tf2_ros import Buffer, TransformListener, TransformException

class RealTimePlotter(Node):
    def __init__(self):
        super().__init__('realtime_plotter')
        
        # --- CẤU HÌNH ---
        self.csv_path = "/home/danh/ros2_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        self.map_frame = "map"       # Hệ quy chiếu gốc (Cố định)
        self.robot_frame = "base_link" # Hệ quy chiếu xe (Di chuyển)
        
        self.history_x = []
        self.history_y = []
        
        # Load Waypoints
        self.ref_x, self.ref_y = self.load_waypoints(self.csv_path)

        # Setup Matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Real-time Tracking (Map Frame Corrected)")
        self.ax.set_xlabel("Map X (m)")
        self.ax.set_ylabel("Map Y (m)")
        self.ax.axis('equal')
        self.ax.grid(True)
        
        # Vẽ Waypoint cố định
        self.ax.plot(self.ref_x, self.ref_y, 'b--', label='Global Waypoints', alpha=0.5)
        self.line_actual, = self.ax.plot([], [], 'r-', label='Robot Position (TF)', linewidth=2)
        self.ax.legend()

        # --- SETUP TF LISTENER (QUAN TRỌNG) ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Thay vì sub odom, ta dùng Timer để query vị trí liên tục
        self.timer = self.create_timer(0.1, self.update_plot) # 10Hz
        
        self.get_logger().info("Plotter bắt đầu lắng nghe TF map -> base_link...")

    def load_waypoints(self, filename):
        wx, wy = [], []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row: continue
                    try:
                        line = row[0].split() if len(row) == 1 else row
                        wx.append(float(line[0]))
                        wy.append(float(line[1]))
                    except: pass
        return wx, wy

    def update_plot(self):
        # --- BƯỚC QUAN TRỌNG: Lấy tọa độ xe trong hệ MAP ---
        try:
            # Tìm transform từ 'map' đến 'base_link'
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rclpy.time.Time()) 
            
            # Lấy tọa độ X, Y chuẩn map
            x = t.transform.translation.x
            y = t.transform.translation.y
            
            self.history_x.append(x)
            self.history_y.append(y)

            # Vẽ lại
            self.line_actual.set_xdata(self.history_x)
            self.line_actual.set_ydata(self.history_y)
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except TransformException as ex:
            # Lỗi này thường xảy ra lúc mới bật, khi chưa có map hoặc chưa định vị xong
            # self.get_logger().warn(f'Không thể lấy vị trí: {ex}')
            pass

def main(args=None):
    rclpy.init(args=args)
    node = RealTimePlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()