#!/usr/bin/env python3

import rclpy
from rclpy.node import Node


from std_msgs.msg import String


class Ros_Publisher(Node):
    def __init__(self):
        super().__init__("publisher_node")
        
        print("publisher_node initialized ")
        
        self.get_logger().info("publisher_node initialized")
        
        self.timer = self.create_timer(1,self.timer_callback)
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        self.chatter = ""
    def timer_callback(self):
        self.chatter="Taihjhjhjh" 
        self.get_logger().info(f"Xin chao {self.chatter} .")
        msg = String()

        msg.data = self.chatter
        self.publisher_.publish(msg)



def main():
    rclpy.init()
    node = Ros_Publisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()
