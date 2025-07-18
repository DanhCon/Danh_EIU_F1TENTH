#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class KiemTra(Node):
    def __init__(self):
        super().__init__("kiem_tra")
        self.group = ReentrantCallbackGroup()
        self.timer_1 = self.create_timer(5.0, self.timer_callback_1,callback_group= self.group)
        self.timer_2 = self.create_timer(5.0, self.timer_callback_2,callback_group= self.group)

    def timer_callback_1(self):
        self.get_logger().info("timer_callback_1")

    def timer_callback_2(self):
        self.get_logger().info("timer_callback_2")

def main(args = None):
        rclpy.init(args=args)
        node = KiemTra()
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()