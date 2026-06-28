#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
import math


class Simple_TurtlessimKinematics(Node):
    def __init__(self):
        super().__init__("simple_turtlesim_kinematics")

        self.turtle1_pose_sub_ = self.create_subscription(Pose, "/turtle1/pose", self.turtle1PoseCallBack, 10)
        self.turtle2_pose_sub_ = self.create_subscription(Pose, "/turtle2/pose", self.turtle2PoseCallBack, 10)

        self.last_turtle1_pose_ = Pose()

        self.last_turtle2_pose_ = Pose()

    def turtle1PoseCallBack(self, msg):
        self.last_turtle1_pose_ =  msg

    def turtle2PoseCallBack(self,msg):
        self.last_turtle2_pose_ = msg
        Tx = self.last_turtle1_pose_.x -self.last_turtle2_pose_.x
        Ty = self.last_turtle1_pose_.y - self.last_turtle2_pose_.y


        theta_rad = self.last_turtle2_pose_.theta - self.last_turtle1_pose_.theta


        theta_deg =100*theta_rad/3.14

        self.get_logger().info("""\n
                    Translation Vector turtle1 -> turtle2 \n
                    tx: %f \n
                    ty: %f \n
                    Rotation Matrix turtle1 -> turtle2 \n
                    theta(rad) : %f\n
                    theta(deg): %f\n
                    |%f      %f|\n
                    |%f      %f|\n
"""%(Tx,Ty,theta_rad, theta_deg, math.cos(theta_rad), -math.sin(theta_rad),
     math.sin(theta_rad), math.cos(theta_rad)))
def main():
    rclpy.init()
    simple_turtlesim_kinematics = Simple_TurtlessimKinematics()
    rclpy.spin(simple_turtlesim_kinematics)
    simple_turtlesim_kinematics.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()