#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from scipy.spatial import KDTree 
from tf2_ros import Buffer, TransformListener, TransformException

class PathAnalyzer(Node):
    def __init__(self):
        super().__init__('path_analyzer_node')
        
        # --- CẤU HÌNH ---
        self.csv_path = "/home/danh/ros2_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        self.map_frame = "map"
        self.robot_frame = "base_link"
        
        # Dữ liệu thực tế
        self.actual_x = []
        self.actual_y = []
        
        # Load và chuẩn bị dữ liệu tham chiếu
        self.ref_x, self.ref_y = self.load_waypoints(self.csv_path)
        
        if not self.ref_x:
            self.get_logger().error("Không load được Waypoints!")
            return

        # TẠO KDTREE
        self.waypoints_np = np.column_stack((self.ref_x, self.ref_y))
        self.tree = KDTree(self.waypoints_np)

        # --- SETUP TF LISTENER (Thay cho Odom Sub) ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Dùng Timer để lấy mẫu dữ liệu (Sample Rate: 20Hz - đủ mịn để phân tích)
        self.timer = self.create_timer(0.05, self.record_position)
        
        self.get_logger().info("System Ready. Waiting for TF... Press Ctrl+C to analyze.")

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

    def record_position(self):
        # Hàm này chạy liên tục 20 lần/giây để lấy vị trí chuẩn Map
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rclpy.time.Time())
            
            x = t.transform.translation.x
            y = t.transform.translation.y
            
            self.actual_x.append(x)
            self.actual_y.append(y)
            
        except TransformException:
            # Lúc mới khởi động có thể chưa có TF, bỏ qua không báo lỗi để đỡ spam
            pass

    def calculate_errors(self):
        if not self.actual_x:
            return None, None, None

        actual_path = np.column_stack((self.actual_x, self.actual_y))
        
        # Tính khoảng cách đến điểm gần nhất trên quỹ đạo mẫu
        distances, _ = self.tree.query(actual_path)
        
        rmse = np.sqrt(np.mean(distances**2))
        max_error = np.max(distances)
        mean_error = np.mean(distances)
        
        return distances, rmse, max_error

    def plot_analysis(self):
        self.get_logger().info("Đang tính toán số liệu...")
        errors, rmse, max_err = self.calculate_errors()
        
        if errors is None:
            self.get_logger().warn("Chưa thu thập được dữ liệu (Check TF/AMCL)!")
            return

        # Setup biểu đồ
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # --- ĐỒ THỊ 1: MAP ---
        ax1.set_title(f"Trajectory Analysis (Map Frame)\nRMSE: {rmse:.4f} m | Max Error: {max_err:.4f} m")
        ax1.plot(self.ref_x, self.ref_y, 'k--', label='Reference (Waypoints)', alpha=0.5)
        
        # Scatter plot với màu sắc thể hiện độ lỗi
        sc = ax1.scatter(self.actual_x, self.actual_y, c=errors, cmap='plasma', s=5, label='Actual Path (TF)')
        plt.colorbar(sc, ax=ax1, label='Error (m)')
        
        ax1.axis('equal')
        ax1.legend()
        ax1.grid(True)

        # --- ĐỒ THỊ 2: ERROR ---
        ax2.set_title("Cross-Track Error over Time")
        ax2.plot(errors, 'r-', linewidth=1)
        ax2.axhline(y=rmse, color='b', linestyle='--', label=f'RMSE ({rmse:.3f}m)')
        ax2.fill_between(range(len(errors)), errors, color='red', alpha=0.1)
        
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Error (m)")
        ax2.legend()
        ax2.grid(True)

        print(f"\n===== KẾT QUẢ PHÂN TÍCH =====")
        print(f"Tổng số mẫu: {len(self.actual_x)}")
        print(f"RMSE (Độ lệch chuẩn): {rmse:.4f} m")
        print(f"Max Error: {max_err:.4f} m")
        print(f"=============================\n")
        
        plt.tight_layout()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = PathAnalyzer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.plot_analysis()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()