#!/usr/bin/env python3
import rclpy 
from rclpy.node import Node
from std_msgs.msg import String

class MiniSub(Node):
    def __init__(self):
        super().__init__("mini_sub")

        self.sub = self.create_subscription(String,'topic_helo', self.listener_callback, 10)
        self.sub 
    def listener_callback(self, msg):

        self.get_logger().info(f'I heard: "{msg.data}"')

def main (args = None):
    rclpy.init(args=args)
    mini_sub = MiniSub()

    rclpy.spin(mini_sub)

    mini_sub.destroy_node()

if __name__ == '__main__':
    main()
