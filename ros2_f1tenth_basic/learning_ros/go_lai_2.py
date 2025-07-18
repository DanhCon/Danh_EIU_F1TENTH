#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor


from rosbasic_msgs.srv import AddTwoInts


class SimpleServer(Node):
    def __init__(self):
        super().__init__("service_server")
        self.group = ReentrantCallbackGroup()

        self.service_ = self.create_service(AddTwoInts, "distance", self.serviceCallback,callback_group= self.group)

        self.get_logger().info("service distance ok readly") 


    def serviceCallback(self, res,rep):
        self.get_logger().info("nhan request , ")
        rep.dis = res.x1 + res.y1
        rep.ok = "da hoan thanh"
        return rep
def main(args = None):
        rclpy.init(args=args)
        node = SimpleServer()
        executor = MultiThreadedExecutor(num_threads=1)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
