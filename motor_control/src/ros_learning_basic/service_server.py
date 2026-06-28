#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MiniService(Node):
    def __init__(self):
        super().__init__('mini_service')


        self.srv = self.create_service(AddTwoInts,"add_two_inits", self.add_two_ints_callback)
        self.get_logger().info("ready to add two ints. ")

    def add_two_ints_callback(self,request,response):
        response.sum = request.a + request.b

        self.get_logger().info(f"incoming request\na: {request.a} b: {request.b}")
        self.get_logger().info(f"sending back response: {response.sum}")

        return response
    
def main(args = None):
    rclpy.init(args=args)
    mini_service = MiniService()
    rclpy.spin(mini_service)
    rclpy.shutdown()

if (__name__ == '__main__'):
    main()