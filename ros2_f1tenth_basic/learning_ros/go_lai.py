#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rosbasic_msgs.srv import AddTwoInts

class distance(Node):
    def __init__(self):
        super().__init__('distace')

        
        
        
        self.declare_parameter("x1",1.0)
        self.declare_parameter("y1",0.0)

        x1 = self.get_parameter("x1").value
        y1 = self.get_parameter("y1").value

        self.get_logger().info(f"using param: x1 ={x1}, y1= {y1}")

        # Tạo client và đợi server sẵn sàng
        
        self.client = self.create_client(AddTwoInts, 'distance')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('⏳ Đợi service "distance"...')


        # Tạo request
        self.req = AddTwoInts.Request()
        self.req.x1 = x1
        self.req.y1 = y1

        # Gửi bất đồng bộ
        self.future = self.client.call_async(self.req) # khi service phản hồi lại thì 
        self.future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        try:
            result = future.result()
            self.get_logger().info(f'✅ Kết quả: {self.req.x1} + {self.req.y1} = {result.dis}')
        except Exception as e:
            self.get_logger().error(f'❌ Gọi service thất bại: {e}')

def main():
    rclpy.init()
    node = distance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
