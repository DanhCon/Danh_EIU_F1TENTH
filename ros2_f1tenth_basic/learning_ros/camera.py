import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class JetsonImageViewer(Node):
    def __init__(self):
        super().__init__('jetson_image_viewer')
        self.subscriber = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.frame_skip = 2   # xử lý mỗi 2 frame
        self.frame_count = 0
        self.get_logger().info('🟢 Subscribed to /camera/camera/color/image_raw')

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return  # bỏ qua frame này

        try:
            # Chuyển từ ROS Image sang OpenCV
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Giảm kích thước ảnh nếu quá lớn
            resized = cv2.resize(img, (640, 480))  # bạn có thể chỉnh về (320,240) nếu cần nhẹ hơn

            # Hiển thị ảnh
            cv2.imshow("Jetson Camera View", resized)
            cv2.waitKey(10)  # thời gian chờ để tránh lag

        except Exception as e:
            self.get_logger().error(f"Lỗi khi xử lý ảnh: {e}")

def main(args=None):
    rclpy.init(args=args)
    viewer = JetsonImageViewer()
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    viewer.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
