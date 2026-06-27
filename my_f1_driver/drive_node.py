import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

class SimpleDriver(Node):
    def __init__(self):
        super().__init__('simple_driver_node')
        
        # Tạo Publisher gửi tín hiệu vào topic '/drive'
        # Topic này là nơi xe lắng nghe lệnh điều khiển
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # Tạo timer để gửi lệnh liên tục (0.05s một lần)
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("Simple Driver Node has been started!")

    def timer_callback(self):
        # Tạo bản tin Ackermann (Kiểu tin nhắn đặc thù cho xe hơi)
        msg = AckermannDriveStamped()
        
        # Tốc độ (m/s)
        msg.drive.speed = -0.1  
        
        # Góc lái (radian) - Dương là rẽ trái, Âm là rẽ phải
        msg.drive.steering_angle = 0.0 
        
        # Gửi tin nhắn đi
        self.publisher_.publish(msg)
        # self.get_logger().info(f'Publishing Speed: {msg.drive.speed}')

def main(args=None):
    rclpy.init(args=args)
    node = SimpleDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()