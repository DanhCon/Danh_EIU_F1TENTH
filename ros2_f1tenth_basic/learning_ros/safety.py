#!/usr/bin/env python3
import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import LaserScan


class docthongtin(Node):
    def __init__(self):
        super().__init__('docthongtin')
        self.sub1 = self.create_subscription(LaserScan,'/scan',self.scan_callback,10)


    def scan_callback(self,msg:LaserScan):
        self.x = msg.angle_min
        self.y = msg.angle_max
        self.range_min = msg.range_min
        self.range_max = msg.range_max

        print(" angle_min:  ",self.x)
        print(" angle_max :  ",self.y)
        print(" range_min :  ",self.range_min)
        print(" range_max :  ",self.range_max)
    
def main(args = None):
    rclpy.init(args = args)
    docthongtin1 = docthongtin()
    rclpy.spin(docthongtin1)
    docthongtin1.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()