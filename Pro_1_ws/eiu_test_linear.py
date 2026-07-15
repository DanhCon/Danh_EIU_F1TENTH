#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from zlac8015d_driver import ZLAC8015D_Driver
from nav_msgs.msg import Odometry
import time
import math
class EIU_TestLinear(Node):
    def __init__(self):
        super().__init__('eiu_test_linear_node')
        
        self.target_distance = 0.85 
        

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # PUBLISHER gui cmd_vel
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.x_pre = None
        self.y_pre = None
        self.accumulated_dist = 0.0
        self.is_running = True

    def odom_callback(self, msg:Odometry):
        cmd = Twist()       
        if self.x_pre is None or self.y_pre is None:
            self.x_pre = msg.pose.pose.position.x
            self.y_pre = msg.pose.pose.position.y
            return
        self.dist = math.sqrt((msg.pose.pose.position.x - self.x_pre)**2 + (msg.pose.pose.position.y - self.y_pre)**2)
        remaining = self.target_distance - self.dist
        print(f"\rXe dang di thang: {self.dist:.3f} m / {self.target_distance:.3f} m   ", end="", flush=True)
        if remaining <= 0 :
            self.get_logger().info("da dung ")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif remaining < 0.1:
            min_vel = 0.05
            cmd.linear.x = min_vel + (0.15 - min_vel) * (remaining / 0.1)
            cmd.angular.z = 0.0
        else:
            cmd.linear.x = 0.15
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)


        
    def stop(self):
        t = Twist()
        t.linear.x = 0.0
        self.cmd_pub.publish(t)
        self.destroy_node()
        rclpy.shutdown()

def main():
    rclpy.init()
    node = EIU_TestLinear()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        
        
        node.stop()

if __name__ == '__main__':
    main()
