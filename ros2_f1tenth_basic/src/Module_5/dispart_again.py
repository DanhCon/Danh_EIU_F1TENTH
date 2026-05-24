#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import qos_profile_sensor_data


class BasicDispartiyExtendeer(Node):
    def __init__(self):
        super().__init__('basic_disparity_node')

        self.fov_min = math.radians(-129)
        self.fov_max = math.radians(120)

        self.car_width = 0.6
        self.disparity_threshold = 0.07

        self.safe_dist = 0.4
        self.prev_angle = 0.0

        self.SMOOTH_ALPHA = 0.35
        self.MAX_GAP_CENTER_ANGLE = math.radians(30.0)
        self.BRAKE_DIST = 0.5
        self.DANGER_DIST = 1.0
        self.DANGER_SPEED = 0.5  
        self.ANGLE_DEADZONE = math.radians(5.0)
        self.FRONT_CONE_DEG = 30.0


        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback)
        self.odom_sub = self.create_subscription(Odometry,'ego_racer.odom',self.odom_callback)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10) # <-- THIẾU DÒNG NÀY
    def preprocess_lidar(self,ranges, max_dist):
        window_size = 4
        ranges_after_fillter = []
        for r in ranges:
            if not math.isinf(r) and not math.isnan(r):
                val_new = min(r,max_dist) # limit khoang cach toi da 
            else:
                val_new = max_dist
            ranges_after_fillter.append(val_new)


        smooth = [0.0]*len(ranges_after_fillter) # tạo mảng mới để lưu giá trị đã được làm mượt


        for i in range(len(ranges_after_fillter)):
            start = max(0, i - window_size//2)
            end = min(len(ranges_after_fillter), i + window_size //2 +1 )
            smooth[i] = sum(ranges_after_fillter[start:end]) / (end - start)
        return smooth
    def extend_dispartities(self, ranges, angle_increment):
        original = ranges.copy()
        filtered = ranges.copy()

        CLOSE_DIST = 2.0
        FAR_MULT = 4.0

        for i in range(len(original) -1 ):
            disparity = abs(original[i] - original[i+1])
            near_dist = min(original[i] , original[i+1])

            threshold = self.disparity_threshold
            if near_dist < CLOSE_DIST:
                threshold = self.disparity_threshold
            else:
                threshold = self.disparity_threshold * FAR_MULT
            if disparity > threshold:
                bubble_ang = 2*math.atan(self.car_width/ (2*max(near_dist, 0.1)))
                bubble_rays = int(bubble_ang/angle_increment)

                if original[i] > original[i + 1]: # tia ben phai longer-> phong to ben phai
                    start = i + 1
                    end = min(len(filtered), i + 1 + bubble_rays)

                    for j in range(start,end):
                        if original[j] > original[start]: # tai sao
                            filtered[j] = 0.0
                else:
                    start = max(0, i - bubble_rays)
                    end = i+ 1
                    for j in range(start,end):
                        if original[j] > original[end]: # tai sao
                            filtered[j] = 0.0
        return filtered
    def find_best_gap(self,ranges, angle_min, angle_increment):
        best_score = -1.0
        best_start = 0
        best_end = 0

        curr_start = -1
        curr_len = 0 
        def evaluate_gap(gap_start,gap_end):
            nonlocal best_score, best_start, best_end
            length = gap_end - gap_start +1
            if length < 3: return

            center_idx = (gap_end+ gap_start)//2
            center_angle = angle_min + center_idx*angle_increment # goc thu te

            if abs(center_angle) > self.MAX_GAP_CENTER_ANGLE: # tâm gap lệch quá nhiều so với tia chính giữa -> không using this gap
                return 
            
            score = length*(math.cos(center_angle)**2) # đánh giá gap dựa trên độ rộng và độ lệch góc, ưu tiên các gap rộng và gần với hướng thẳng trước mặt
            if score > best_score:
                best_score = score
                best_start = gap_start
                best_end = gap_end
        for i,r in enumerate(ranges):
            if r > self.safe_dist:
                if curr_start == -1:
                    curr_start = i
                    
                curr_len +=1
            else:
                if curr_len > 0:
                    evaluate_gap(curr_start,curr_start + curr_len-1)
                curr_start,   curr_len = -1   ,   0
        if curr_len > 0:
            evaluate_gap(curr_start, curr_start + curr_len - 1)
        return best_start, best_end, best_score > 0

    def find_best_point(self, start_idx, end_idx, ranges):
        if start_idx >= end_idx:
            return (start_idx + end_idx)//2  # tránh chương trình bị crack khi không có gap hợp lệ, trả về chỉ số trung tâm của khoảng đã cho
        sub_gap = ranges[start_idx:end_idx + 1]
        max_val = max(sub_gap)
        threshold = max_val* 0.45

        best_start = best_end = 0
        max_width = 0
        car_start = -1
        car_width = 0


        for i, val in enumerate(sub_gap ):
            if val >= threshold:
                if car_start == -1: car_start = i
                car_width +=1
            else:
                if car_width > max_width:
                    max_width, best_start, best_end = car_width, car_start, i - 1
                car_start, car_width = -1, 0
        if car_width > max_width:         # dành cho trường hợp gap kết thúc bằng một hành lang rộng
            best_start, best_end = car_start, len(sub_gap) - 1

    # Trả về chỉ số trung tâm của hành lang tốt nhất (ánh xạ về mảng gốc)
        return start_idx + (best_start + best_end) // 2
    def lidar_callback(self, data: LaserScan):
        # 1. Tính toán chỉ số để cắt mảng dữ liệu theo FOV [-120°, 120°]
        s_idx = int((self.fov_min - data.angle_min) / data.angle_increment)
        e_idx = min(int((self.fov_max - data.angle_min) / data.angle_increment), len(data.ranges) - 1) 

        # Trích xuất đoạn dữ liệu nằm trong FOV quan tâm
        raw_ranges = list(data.ranges[s_idx:e_idx])
        
        # 2. Chuỗi tiền xử lý dữ liệu (Perception Stage)



        ranges          = self.preprocess_lidar(raw_ranges, 5.0)  # 5.0 là giới hạn max là 5 m để tránh ảnh hưởng của các giá trị cực lớn hoặc lỗi đo đạc

        # 3. Kiểm tra an toàn cứng (Hard Safety Layer) trước khi quy hoạch đường đi
        front_clear = self.check_front_clearance(ranges, self.fov_min, data.angle_increment)

        if front_clear < self.BRAKE_DIST:
            # Nếu tường ở ngay trước mặt (< 0.5m) -> Phanh khẩn cấp, giữ nguyên góc lái cũ
            self.get_logger().warn(f"🛑 EMERGENCY BRAKE: front_clear={front_clear:.2f}m < {self.BRAKE_DIST}m")
            self.publish_drive(self.prev_angle, 0.0)
            return

        # 4. Thực thi thuật toán lõi hình học (Planning Stage)
        # Mở rộng bước nhảy tạo Safety Bubble xung quanh các cạnh vật cản
        ranges = self.extend_dispartities(ranges, data.angle_increment)  # giúp loại bỏ các gap hẹp không đủ rộng cho xe đi qua, tạo ra các khoảng trống an toàn hơn để lựa chọn

        # Tìm khoảng trống (Gap) tốt nhất hướng về phía trước
        gap_s, gap_e, found = self.find_best_gap(ranges, self.fov_min, data.angle_increment) # tìm gap tốt nhất dựa trên độ rộng và vị trí, ưu tiên các gap rộng và gần với hướng tia chính giữa xe

        if not found:
            # Nếu bị vây kín hoặc không tìm được đường đi hợp lệ -> Dừng xe an toàn
            self.get_logger().warn("⚠️ Không có gap hợp lệ hướng về phía trước → DỪNG")
            self.publish_drive(0.0, 0.0)
            return

        # Tìm điểm sâu và thoáng nhất bên trong Gap đã chọn
        best_idx  = self.find_best_point(gap_s, gap_e, ranges)  # tìm điểm tốt nhất trong gap đã chọn, ưu tiên các điểm có nằm trong hành lang rộng và có khoảng cách lớn hơn để đảm bảo an toàn khi lái qua gap đó
                                                                # lấy chỉ số của điểm tốt nhất trong mảng đã cắt, sau đó ánh xạ về chỉ số gốc bằng cách cộng với start_idx
        raw_angle = self.fov_min + best_idx * data.angle_increment # không sài angle_min mà fov_min vì đã cắt mảng rồi, chỉ cần dùng fov_min làm gốc để tính góc tương ứng với best_idx trong mảng đã cắt là được
                                                                    # sài angle_min sẽ bị sai vì đã cắt mảng rồi, nếu dùng angle_min sẽ tính sai góc tương ứng với best_idx trong mảng đã cắt, dẫn đến lái lệch hướng mong muốn
        # Áp dụng Deadzone: Nếu góc lái quá nhỏ, cho xe đi thẳng để tránh rung lắc servo
        if abs(raw_angle) < self.ANGLE_DEADZONE: # giảm nhiễu và tránh lái quá nhạy khi góc lái nhỏ, giúp xe đi thẳng hơn khi không cần thiết phải điều chỉnh nhiều
            raw_angle = 0.0

        # 5. Bộ lọc làm mượt tín hiệu điều khiển (Control Stage)
        smooth_angle    = self.SMOOTH_ALPHA * raw_angle + (1 - self.SMOOTH_ALPHA) * self.prev_angle  # làm mượt nhờ kết hợp góc lái mới với góc lái trước đó
        self.prev_angle = smooth_angle     # cập nhật góc lái trước đó để sử dụng trong lần gọi callback tiếp theo, giúp tạo hiệu ứng làm mượt liên tục qua các lần đo và tính toán

        # 6. Chiến lược thiết lập tốc độ tối ưu (Speed Profile)
        if front_clear < self.DANGER_DIST:
            speed = self.DANGER_SPEED  # Chế độ bò dò đường: 0.5 m/s
        elif abs(smooth_angle) < math.radians(10.0):
            speed = 5.0              # Đường thẳng: Chạy tối đa 5.0 m/s
        elif abs(smooth_angle) < math.radians(20.0):
            speed = 3.0              # Cua vừa: Giảm xuống 3.0 m/s
        else:
            speed = 2.0              # Cua ngặt: Giảm xuống 2.0 m/s để bám đường

        # 7. Gửi lệnh xuống hệ thống lái Ackermann
        self.publish_drive(smooth_angle, speed)
    def check_front_clearance(self, ranges, angle_min, angle_increment):
        """
        Trả về khoảng cách nhỏ nhất trong vùng ±FRONT_CONE_DEG phía trước.
        Dùng để quyết định có cần brake khẩn cấp không.
        """
        cone_rad  = math.radians(self.FRONT_CONE_DEG)
        n         = len(ranges)
        min_dist  = float('inf')
        for i, r in enumerate(ranges):
            angle = angle_min + i * angle_increment
            if abs(angle) <= cone_rad and r > 0.01:
                if r < min_dist:
                    min_dist = r
        return min_dist if min_dist != float('inf') else 5.0
    def publish_drive(self, angle, speed):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = speed
        self.get_logger().info(f"🚗 Lệnh lái: steering_angle={math.degrees(angle):.1f}°, speed={speed:.2f} m/s")
        self.prev_angle = angle
        self.drive_pub.publish(drive_msg)   
    
    def odom_callback(self, data: Odometry):
        # Có thể dùng dữ liệu odometry để cải thiện dự đoán hoặc điều chỉnh chiến lược lái
        pass    
def main(args=None):    
    rclpy.init(args=args)
    node = BasicDispartiyExtendeer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

