#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class EIU_TestAngular(Node):
    def __init__(self):
        super().__init__('eiu_test_angular_node')
        
        self.wheel_base = 0.45
        self.target_angle = 1 * math.pi # 360 độ
        self.current_speed = 2.5 # Tốc độ mong muốn ban đầu
        
        # SUBSCRIBER de lay JointState (Đoạn đường) tinh Odom Tuyet Doi
        self.joint_sub = self.create_subscription(Odometry, 'odom', self.joint_callback, 10)
        
        # PUBLISHER gui cmd_vel
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.start_l = None
        self.start_r = None
        self.accumulated_angle = 0.0
        self.is_running = True
        
        self.get_logger().info("=== Kiem chung phuong phap EIU ===")
        
        
        # Timer de lien tuc gui cmd_vel 
        self.timer = self.create_timer(0.1, self.timer_callback)

    def joint_callback(self, msg):

        
        if not self.is_running: return
        if len(msg.position) < 2: return
        
        l_curr = msg.position[0]
        r_curr = msg.position[1]
        
        if self.start_l is None:
            self.start_l = l_curr
            self.start_r = r_curr
            return
            
        delta_l = l_curr - self.start_l
        delta_r = r_curr - self.start_r
        
        self.accumulated_angle = abs(delta_r - delta_l) / self.wheel_base
        print(f"\rXe dang xoay (EIU Style): {math.degrees(self.accumulated_angle):.1f} do / 360.0 do   ", end="", flush=True)
        
        if self.accumulated_angle >= self.target_angle:
            print()
            self.get_logger().info("Hoan thanh xoay 360 do! Dung xe.")
            self.stop()

    def timer_callback(self):
        if not self.is_running: return
        
        # HÃM TỐC KHI GẦN TỚI ĐÍCH (Chống trôi quán tính 200ms của Motor)
        remaining = self.target_angle - self.accumulated_angle
        
        if remaining < 0.2:
            # Giảm tốc độ từ từ xuống rất chậm (0.1 rad/s) để dừng chính xác
            self.current_speed = 0.1 + (0.4 - 0.1) * (max(0.0, remaining) / 0.2)
            
        t = Twist()
        t.angular.z = float(self.current_speed)
        self.cmd_pub.publish(t)
        
    def stop(self):
        self.is_running = False
        t = Twist()
        t.angular.z = 0.0
        self.cmd_pub.publish(t)
        self.destroy_node()
        rclpy.shutdown()

def main():
    rclpy.init()
    node = EIU_TestAngular()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()

if __name__ == '__main__':
    main()
