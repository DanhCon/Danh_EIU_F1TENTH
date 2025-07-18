#!/usr/bin/env python3
import rclpy 
from rclpy.node import Node
from rosbasic_msgs.srv import BaiTap


class SimpleServiceCLient(Node):

    def __init__(self):
        super().__init__("Simple_service_client")
        
        self.declare_parameter("distance_odom" , 0.0)
        self.declare_parameter("distance_lidar", 0.0)

        distance_odom = self.get_parameter("distance_odom").value
        distance_lidar = self.get_parameter("distance_lidar").value

        self.get_logger().into()

        self.client_ = self.create_client(BaiTap,"BaiTap") # hai chu baitap khacs gif nhau

        while not self.client_.wait_for_service(timeout_sec= 1.0):
            self.get_logger().into("dkjfkdjf")
        self.req_ = BaiTap.Request()
        self.req_.distance_odom = distance_odom
        self.req_.distance_odom = distance_lidar

        self.future = self.client_.call_async(self.req_)
        self.future.add_done_callback(self.responseCallback) # DE LAM GI 
    def responseCallback(self,future):
            try:
                 response = future.result()
                 self.get_logger().info("")
            except Exception as e:
                 self.get_logger().error(e)
def main():
     rclpy.init()
     client_node = SimpleServiceCLient()
     rclpy.spin(client_node)
     client_node.destroy_node()
     rclpy.shutdown()
if __name__ == "__main__":
     main()
                

