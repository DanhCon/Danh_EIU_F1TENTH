#!/usr/bin/env python3
import numpy as np
import math
import json
import os

def generate_figure_8(a=2.0, b=1.0, num_points=200, v_target=0.3):
    """ Sinh quỹ đạo Hình số 8 (Figure-8) """
    t = np.linspace(0, 2 * np.pi, num_points)
    x = a * np.sin(t)
    y = b * np.sin(2 * t)
    
    # Tính đạo hàm để tìm vận tốc và góc yaw
    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    yaw = np.arctan2(dy, dx)
    
    dyaw = np.gradient(yaw, t)
    ds = np.sqrt(dx**2 + dy**2)
    w = dyaw / np.maximum(ds, 1e-6) * v_target

    waypoints = []
    for i in range(num_points):
        waypoints.append({
            "index": i,
            "time": round(i * 0.1, 2),
            "x": round(float(x[i]), 4),
            "y": round(float(y[i]), 4),
            "yaw_rad": round(float(yaw[i]), 4),
            "yaw_deg": round(math.degrees(float(yaw[i])), 2),
            "v_linear": v_target,
            "w_angular": round(float(w[i]), 3)
        })
    return waypoints

def generate_square_waypoints(side_length=2.0, num_points_per_side=50, v_target=0.3):
    """ Sinh quỹ đạo Hình Vuông bo tròn góc """
    # Tạo các điểm góc vuông
    corners = [(0, 0), (side_length, 0), (side_length, side_length), (0, side_length), (0, 0)]
    
    waypoints = []
    total_idx = 0
    
    for i in range(len(corners) - 1):
        x1, y1 = corners[i]
        x2, y2 = corners[i+1]
        
        dx = x2 - x1
        dy = y2 - y1
        segment_yaw = math.atan2(dy, dx)
        
        xs = np.linspace(x1, x2, num_points_per_side, endpoint=False)
        ys = np.linspace(y1, y2, num_points_per_side, endpoint=False)
        
        for j in range(num_points_per_side):
            waypoints.append({
                "index": total_idx,
                "time": round(total_idx * 0.1, 2),
                "x": round(float(xs[j]), 4),
                "y": round(float(ys[j]), 4),
                "yaw_rad": round(segment_yaw, 4),
                "yaw_deg": round(math.degrees(segment_yaw), 2),
                "v_linear": v_target,
                "w_angular": 0.0
            })
            total_idx += 1
            
    return waypoints

def main():
    print("=== CÔNG CỤ TẠO QUỸ ĐẠO TỰ ĐỘNG ===")
    print("1. Tạo quỹ đạo Hình số 8 (Figure-8)")
    print("2. Tạo quỹ đạo Hình vuông (Square 2m x 2m)")
    choice = input("Lựa chọn kiểu quỹ đạo muốn tạo (1 hoặc 2): ").strip()

    if choice == '2':
        waypoints = generate_square_waypoints(side_length=2.0)
        shape_name = "Hình Vuông"
    else:
        waypoints = generate_figure_8(a=2.0, b=1.0)
        shape_name = "Hình số 8"

    filename = 'trajectory.json'
    cwd = os.getcwd()
    filepath = os.path.join(cwd, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "total_points": len(waypoints),
                "type": shape_name,
                "created_by": "Mathematical Generator"
            },
            "waypoints": waypoints
        }, f, indent=4)

    print(f"\n✅ ĐÃ TẠO THÀNH CÔNG QUỸ ĐẠO [{shape_name}]!")
    print(f"📍 Tổng số điểm: {len(waypoints)} điểm")
    print(f"📁 File đã lưu tại: {filepath}")
    print("\n👉 Bây giờ bạn có thể bật `lqr_trajectory_follower_node.py` để xe chạy bám theo ngay!")

if __name__ == '__main__':
    main()