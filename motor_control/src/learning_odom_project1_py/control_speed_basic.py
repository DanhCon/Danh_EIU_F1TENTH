#!/usr/bin/env python3 



import rclpy

from rclpy.node import Node 
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray
import numpy as np

class SimpleController(Node):
    def __init__(self):
        super().__init__("simple_controller")

        self.declare_parameter("wheel_radius", 0.033)
        self.declare_parameter("wheel_separation", 0.17)

        self.wheel_radius_ = self.get_parameter("wheel_radius").get_parameter_value().double_value
        self.wheel_separation_ = self.get_parameter("wheel_separation").get_parameter_value().double_value


        self.get_logger().info("Using wheel_radius %f" % self.wheel_radius_)
        self.get_logger().info("Using wheel_separation %f" % self.wheel_separation_)


        self.wheel_cmd_pub_ = self.create_publisher("simple_velocity_controller/commands", 10)

        self.vel_sub_ = self.create_subscription("bumperbot_controller/cmd_vel", self.velCallback, 10)

        self.speed_conversion_ = np.array([self.wheel_radius_/2, self.wheel_radius_/2], 
                                          [self.wheel_radius_/self.wheel_separation_, -self.wheel_radius_/self.wheel_separation_])
        
        self.get_logger().info("the conversion matrix is %s" %self.speed_conversion_)


    def velCallback(self, msg):


        robot_speed = np.array([[msg.twist.linear.x], 
                               [msg.twist.angular.z]])
        

        wheel_speed = np.matmul(np.linalg.inv(self.speed_conversion_))

        wheel_speed_msg = Float64MultiArray()
        wheel_speed_msg = [wheel_speed[1,0], wheel_speed[0,0]]
        self.wheel_cmd_pub_.publish(wheel_speed_msg)

def main():
    rclpy.init()
    simple_controller = SimpleController()
    rclpy.spin(simple_controller)
    simple_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()








