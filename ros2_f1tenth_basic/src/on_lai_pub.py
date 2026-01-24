#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MiniPub(Node):
    def __init__(self):
        super().__init__("mini_pub")

        self.pub_ =self.create_publisher(String, 'topic_helo', 10)

        time_period = 0.5

        self.timer = self.create_timer(time_period, self.timer_callback)

        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f"hellllo: {self.i}"

        self.pub_.publish(msg)

        self.get_logger().info(f'Publishing: "{msg.data}"')

        self.i +=1

def main(args = None):
    rclpy.init(args=args)

    mini_pub = MiniPub()

    rclpy.spin(mini_pub)

    mini_pub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from example_interfaces.srv import AddTwoInts 

# class AsyncClient(Node):
#     def __init__(self):
#         super().__init__('async_client')
        
#         self.declare_parameter("so_a", 4)
#         self.declare_parameter("so_b", 5)


#         self.a = self.get_parameter("so_a").value
#         self.b = self.get_parameter("so_b").value

#         self.client_ = self.create_client(AddTwoInts, "AddTwoInts")
        
 
#         while not self.client_.wait_for_service(timeout_sec=1.0):
#             self.get_logger().warn("Đang tìm Service... vui lòng bật Server lên!")


#         self.send_request()

#     def send_request(self):
#         request = AddTwoInts.Request()
#         request.a = self.a
#         request.b = self.b

#         self.get_logger().info(f"Đang gửi yêu cầu tính: {self.a} + {self.b} ...")

#         self.future_ = self.client_.call_async(request)
        

#         self.future_.add_done_callback(self.handle_response)

#     def handle_response(self, future):
#         try:
#             response = future.result()
#             self.get_logger().info(f"--> KẾT QUẢ TỪ SERVER: {response.sum}")
#         except Exception as e:
#             self.get_logger().error(f"Gọi service thất bại: {e}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = AsyncClient()
    

#     rclpy.spin(node)
    
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

