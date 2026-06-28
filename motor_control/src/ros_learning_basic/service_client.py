#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts 
from nav_msgs.msg import Odometry

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class Client(Node):
    def __init__(self):
        super().__init__('async_client')
        self.group = ReentrantCallbackGroup()

        self.sub_odom = self.create_subscription(Odometry, "/odom", self.timer_callback, 10, callback_group=self.group)
        
        
        self.declare_parameter("so_a", 4)
        self.declare_parameter("so_b", 5)


        self.a = self.get_parameter("so_a").value
        self.b = self.get_parameter("so_b").value

        self.client_ = self.create_client(AddTwoInts, "AddTwoInts",callback_group=self.group)
        
 
        while not self.client_.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Đang tìm Service... vui lòng bật Server lên!")


        self.send_request()

    def send_request(self):
        request = AddTwoInts.Request()
        request.a = self.a
        request.b = self.b

        self.get_logger().info(f"Đang gửi yêu cầu tính: {self.a} + {self.b} ...")

        self.future_ = self.client_.call_async(request)
        

        self.future_.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"--> KẾT QUẢ TỪ SERVER: {response.sum}")
        except Exception as e:
            self.get_logger().error(f"Gọi service thất bại: {e}")
            
    def timer_callback(self, msg:Odometry ):


       self.orient_w=msg.pose.pose.orientation.w
       self.linear_x=msg.twist.twist.linear.x


       self.get_logger().info(f"w: {self.orient_w}")
       self.get_logger().info(f"linear x: {self.linear_x}")            

def main():
    rclpy.init()
    try:
        talker = Client()
        # parallel to the ones in DoubleTalker however.
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(talker)
        try:
            executor.spin()
        finally:
            talker.run = False
            executor.shutdown()
            talker.destroy_node()
    finally:
        rclpy.shutdown()
if __name__ == '__main__':
    main()