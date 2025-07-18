#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import math
import time
from rosbasic_msgs.srv import BaiTap

class SimpleServiceServer(Node):
    def __init__(self):
        super().__init__("simple_service_server")
        self.group = ReentrantCallbackGroup()

        # Khai báo Service, Publisher, Subscriber với callback group
        self.service_ = self.create_service(BaiTap, "BaiTap", self.serviceCallback, callback_group=self.group)
        self.publisher_ = self.create_publisher(AckermannDriveStamped, "/drive", 10, callback_group=self.group)
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.lisener, 10, callback_group=self.group)
        self.sub_lidar = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10, callback_group=self.group)

        self.current_pos = None
        self.initialized = None
        self.distance_lidar = None
        self.distance_odom = None
        
        

        self.get_logger().info("Service BaiTap Ready")
        # print(self.distance_odom)

    def serviceCallback(self, request, response):
        self.get_logger().info(" Nhận request, bắt đầu di chuyển")
        self.status= True
        self.initialized=None

        cmd = AckermannDriveStamped()
        cmd.drive.steering_angle = 0.0

        while self.current_pos is None:
            self.get_logger().info("Chưa có dữ liệu odom...")
            # time.sleep(0.1)

        start_pos = self.current_pos.copy()

        while self.status:
            dx = self.current_pos[0] - start_pos[0]
            dy = self.current_pos[1] - start_pos[1]
            self.travel = math.sqrt(dx**2 + dy**2)

            if self.travel >= 20.0:
                self.get_logger().info("Đã đi đủ 20m")
                self.status = False

            if self.distance_lidar is None:
                self.get_logger().info(" Đợi giá trị lidar...")
                # time.sleep(0.1)
                continue

            if self.distance_lidar < 4.0:
                cmd.drive.speed = 0.0
                self.get_logger().info(" Vật cản phát hiện - dừng")
            else:
                cmd.drive.speed = 1.0
                self.get_logger().info(" Đường thông - tiếp tục")

            self.publisher_.publish(cmd)
            #time.sleep(0.1)

        cmd.drive.speed = 0.0
        self.publisher_.publish(cmd)
        self.get_logger().info(" Dừng hoàn toàn")

        response.result = True
        
        response.message = " Đã hoàn thành di chuyển 20m"
        return response

    def scan_callback(self, scan_msg):
        angle_deg = 0
        angle_rad = math.radians(angle_deg)
        index = int((angle_rad - scan_msg.angle_min) / scan_msg.angle_increment)
        if 0 <= index < len(scan_msg.ranges):
            distance = scan_msg.ranges[index]
            if not math.isnan(distance):
                self.distance_lidar = distance
                print(f"laser: distance {distance}  distance {self.distance_odom}")

    def lisener(self, msg:Odometry):
        self.current_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        if self.initialized is None:
            self.initialized = self.current_pos.copy()
        self.distance_odom = math.sqrt((self.current_pos[0] - self.initialized[0])**2 + (self.current_pos[1] - self.initialized[1])**2)
        print("distance:  ", self.distance_odom)
def main(args=None):
    rclpy.init(args=args)
    node = SimpleServiceServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
