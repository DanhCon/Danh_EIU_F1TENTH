#!/usr/bin/env python3
import rclpy
from rclpy.node import Node 
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class docthongtin(Node):
    def __init__(self):
        super().__init__('docthongtin')
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.lisener, 10)
        self.publishers_ = self.create_publisher(AckermannDriveStamped, '/drive',10 )
    def lisener(self,msg:Odometry):
        run = AckermannDriveStamped()
        run.drive.speed = 0.0
        self.publishers_.publish(run)
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        self.z = msg.pose.pose.orientation.z
        self.w = msg.pose.pose.orientation.w

        print("x:  ", self.x )
        print("y:  ", self.y )
        print("z:  ", self.z )
        print("w:  ", self.w )

def main(args = None):
    rclpy.init(args = args)
    docthongtin1 = docthongtin()
    rclpy.spin(docthongtin1)
    docthongtin1.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

               

