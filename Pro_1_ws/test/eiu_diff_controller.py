#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import numpy as np
import math

class EIU_DiffController(Node):
    def __init__(self):
        super().__init__('eiu_diff_controller_node')
        
        self.wheel_radius = 0.0535
        self.wheel_separation = 0.45
        

        self.max_linear_velocity = 0.6
        self.max_angular_velocity = 0.4
        
        self.vel_sub = self.create_subscription(Twist, 'cmd_vel', self.vel_callback, 10)
        self.wheel_cmd_pub = self.create_publisher(Float64MultiArray, 'eiu/wheel_rotational_vel', 10)
        
        self.get_logger().info(f"Diff Controller (EIU Style) khoi dong. Gioi han goc: {self.max_angular_velocity} rad/s")

    def vel_callback(self, msg):

        linear_vel = np.clip(msg.linear.x, -self.max_linear_velocity, self.max_linear_velocity)
        angular_vel = np.clip(msg.angular.z, -self.max_angular_velocity, self.max_angular_velocity)
        
        # 2. Tính vận tốc từng bánh (m/s)
        right_wheel = linear_vel + (angular_vel * self.wheel_separation / 2.0)
        left_wheel = linear_vel - (angular_vel * self.wheel_separation / 2.0)
        
        # 3. Đổi sang Rad/s rồi sang RPM
        # V = W * R => W = V / R
        left_rpm = (left_wheel / self.wheel_radius) * 60.0 / (2 * math.pi)
        right_rpm = (right_wheel / self.wheel_radius) * 60.0 / (2 * math.pi)
        
        # 4. Gửi mảng RPM đi
        out_msg = Float64MultiArray()
        out_msg.data = [float(left_rpm), float(right_rpm)]
        self.wheel_cmd_pub.publish(out_msg)

def main():
    rclpy.init()
    node = EIU_DiffController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
