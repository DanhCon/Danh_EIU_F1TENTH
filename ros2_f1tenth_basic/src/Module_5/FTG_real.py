#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import LaserScan 
from ackermann_msgs.msg import AckermannDriveStamped
import math

from visualization_msgs.msg import Marker
from rclpy.qos import qos_profile_sensor_data
class PureFTG(Node):
    def __init__(self):
        super().__init__('pure_ftg_node')

        self.range_lidar_min = math.radians(-90.0)
        self.range_lidar_max = math.radians(90.0)
        self.car_width = 0.3
        self.safe_dist = 0.6

        self.max_steer = math.radians(20)
        self.v_max = 1.0
        self.v_min = 1.0
        self.latest_ranges = None
        self.latest_scan_msg = None

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback,10 )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive' , 10)

        self.marker_pub = self.create_publisher(Marker, '/virtual_target_point', 10)

        self.control_timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info("Pure FTG danh node initalized")
    def lidar_callback(self, msg:LaserScan):
        self.latest_scan_msg = msg
        self.latest_ranges = np.array(msg.ranges)


    def compute_ftg_simple(self, ranges:list, scan_msg: LaserScan ) -> tuple:
        start_idx = int((self.range_lidar_min - scan_msg.angle_min) / scan_msg.angle_increment)
        end_idx = int((self.range_lidar_max - scan_msg.angle_min) / scan_msg.angle_increment)
        new_ranges = ranges[start_idx:end_idx]

        for  i in range(len(new_ranges)):
            if new_ranges[i] > scan_msg.range_max:
                new_ranges[i] = scan_msg.range_max
            elif new_ranges[i] == 0.0 or math.isnan(new_ranges[i]):
                new_ranges[i] = 0
        smoothed__ranges  = [0.0]* len(new_ranges)
        for i in range(1, len(new_ranges) -1 ):
            smoothed__ranges[i] = (new_ranges[i-1] + new_ranges[i]+ new_ranges[i+1])/3

        new_ranges= smoothed__ranges



        #####################################################
        min_dist = float('inf')
        min_idx = -1

        for i in range(len(new_ranges)):
            if new_ranges[i] > 0.1 and new_ranges[i] < min_dist:
                min_dist = new_ranges[i]
                min_idx = i

        bubble_angle = math.atan(self.car_width / min_dist)
        num_rays_to_clear = int (bubble_angle/ scan_msg.angle_increment) # goc / do chia nho nhat (90/1) = 90 tia need to clear

        start_bubble = max(0, min_idx  - num_rays_to_clear)
        end_bubble = min(len(new_ranges) - 1 , min_idx + num_rays_to_clear)

        for i in range(start_bubble, end_bubble+1):
            new_ranges[i] = 0.0

    ##################################################################
        max_start = 0 
        max_end = 0 
        max_length = 0 

        current_start = -1
        current_length = 0

        for i in range(len(new_ranges)):
            if new_ranges[i] > self.safe_dist:
                if current_start == -1:
                    current_start = i
                current_length += 1
            else:
                if current_length > max_length:
                    max_length = current_length
                    max_start = current_start
                    max_end = i - 1
                current_start = -1
                current_length = 0
        if current_length> max_length:
            max_start = current_start
            max_end = len(new_ranges) - 1

        if max_length ==0:
            return 0.0, 0.0
        
        target_relative_idx = (max_start +max_end) //2 

        global_taget_idx = start_idx + target_relative_idx
        steering_angle = scan_msg.angle_min + global_taget_idx* scan_msg.angle_increment
        steering_angle = max(-self.max_steer, min(self.max_steer, steering_angle))

        #######################################
        target_relative_idx = (max_start + max_end) // 2 
        global_target_idx = start_idx + target_relative_idx
        
        # Góc nhắm tới khe hở (Theta)
        raw_target_angle = scan_msg.angle_min + global_target_idx * scan_msg.angle_increment
        
        # Khoảng cách từ xe đến cái khe hở đó (Radius)
        # Lưu ý: Dùng mảng ranges nguyên thủy (chưa bị đè bằng 0 ở bong bóng) để lấy khoảng cách thật
        target_distance = ranges[global_target_idx]
        target_x = target_distance * math.cos(raw_target_angle)
        target_y = target_distance * math.sin(raw_target_angle)
        
        # Gọi hàm vẽ Marker (lấy frame_id trực tiếp từ tin nhắn LiDAR để hình không bị trôi)
        self.publish_target_marker(target_x, target_y, scan_msg.header.frame_id)

        if (abs(steering_angle) > math.radians(15.0)):
            speed = self.v_min
        else:
            speed = self.v_max
        return steering_angle, speed
    def control_loop(self):

        if self.latest_ranges is None or self.latest_scan_msg is None:
            return 
        current_ranges = np.copy(self.latest_ranges)
        scan_msg = self.latest_scan_msg

        steering_angle, speed = self.compute_ftg_simple(current_ranges, scan_msg)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering_angle)
        drive_msg.drive.speed = float(speed)

        self.drive_pub.publish(drive_msg)
    def publish_target_marker(self, target_x, target_y, frame_id):
        """ Hàm dùng để vẽ một quả cầu màu xanh lá tại mục tiêu """
        marker = Marker()
        marker.header.frame_id = frame_id # Hệ quy chiếu (Thường là "laser" hoặc "base_link")
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = "ftg_target"
        marker.id = 0
        marker.type = Marker.SPHERE # Hình quả cầu
        marker.action = Marker.ADD

        # Tọa độ Đề-các (X, Y)
        marker.pose.position.x = float(target_x)
        marker.pose.position.y = float(target_y)
        marker.pose.position.z = 0.0 # Nằm bệt trên mặt đất

        # Kích thước quả cầu (0.3 mét = 30cm)
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        # Màu sắc (RGB: Xanh lá cây rực rỡ)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0 # 1.0 là hiển thị đặc, 0.5 là bán trong suốt

        # Phát tín hiệu lên RViz
        self.marker_pub.publish(marker)
def main(args = None):
    rclpy.init(args=args)
    node = PureFTG()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("shuting ")
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()






