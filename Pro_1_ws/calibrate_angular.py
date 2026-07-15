import rclpy
from rclpy.node import Node
import time
import math
from zlac8015d_driver import ZLAC8015D_Driver

class CalibrateAngularNode(Node):
    def __init__(self):
        super().__init__('calibrate_angular_node')

        # === Đã cập nhật các thông số đã hiệu chuẩn ===
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('wheel_radius', 0.0610)         # Đã hiệu chuẩn tuyến tính
        self.declare_parameter('wheel_base', 0.4680)             # Đã cập nhật theo yêu cầu
        self.declare_parameter('cpr', 4096)
        self.declare_parameter('travel_in_1_vong', 0.3830)     # Đã hiệu chuẩn tuyến tính
        
        # Góc mục tiêu: Quay đúng 1 vòng tròn 360 độ (2 * PI)
        self.declare_parameter('test_angle', 2 * math.pi)      
        self.declare_parameter('angular_speed', 2.5)           # rad/s

        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value
        self.wheel_radius = self.get_parameter('wheel_radius').get_parameter_value().double_value
        self.wheel_base = self.get_parameter('wheel_base').get_parameter_value().double_value
        self.cpr = self.get_parameter('cpr').get_parameter_value().integer_value
        self.travel_in_one_rev = self.get_parameter('travel_in_1_vong').get_parameter_value().double_value
        self.test_angle = self.get_parameter('test_angle').get_parameter_value().double_value
        self.angular_speed = self.get_parameter('angular_speed').get_parameter_value().double_value

        self.get_logger().info("=== Bắt đầu hiệu chuẩn góc quay (Angular Calibration) ===")
        self.get_logger().info(f"Mục tiêu: Xoay tại chỗ {math.degrees(self.test_angle):.0f} độ ({(self.test_angle):.4f} rad) với tốc độ {self.angular_speed} rad/s")

        self.driver = ZLAC8015D_Driver(
            port=self.port,
            baudrate=self.baudrate,
            wheel_base=self.wheel_base,
            wheel_radius=self.wheel_radius,
            cpr=self.cpr,
            travel_in_1_vong=self.travel_in_one_rev,
        )
        if not self.driver.init_motor():
            self.get_logger().error('Lỗi kết nối động cơ')
            return 
            
        # Lấy vị trí Odom ban đầu làm mốc (Điểm Đầu)
        l_start, r_start = self.driver.get_wheels_travelled()
        while l_start is None:
            time.sleep(0.1)
            l_start, r_start = self.driver.get_wheels_travelled()

        self.start_l = l_start
        self.start_r = r_start
        
        self.odom_angle_accumulated = 0.0
        self.is_running = True

        # Tần số 10Hz để giảm tải RS485 chống nghẽn
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        if not self.is_running:
            return 
            
        # Đọc Odom hiện tại (Điểm Cuối)
        l_curr, r_curr = self.driver.get_wheels_travelled()
        if l_curr is None or r_curr is None:
            return # Lỗi đọc RS485, bỏ qua chu kỳ này
        
        # Tính khoảng cách đã đi của mỗi bánh
        delta_l = l_curr - self.start_l
        delta_r = r_curr - self.start_r

        # Động học ngược: Tính góc quay dựa trên khoảng cách vi sai
        # Công thức: Theta = (Quãng đường Phải - Quãng đường Trái) / Khoảng cách bánh
        self.odom_angle_accumulated = abs(delta_r - delta_l) / self.wheel_base

        # Đã thêm ký tự \r để in đè mượt mà không bị dính dòng
        print(f"\rXe đã xoay {math.degrees(self.odom_angle_accumulated):.1f} độ / {math.degrees(self.test_angle):.1f} độ    ", end="", flush=True)

        remaining = abs(self.test_angle) - self.odom_angle_accumulated

        if remaining <= 0:
            print()
            self.stop_test()
            return 
            
        target_speed = self.angular_speed
        
        # Giảm tốc độ xoay khi gần đến đích (giảm từ từ xuống 0.05 rad/s khi còn 0.2 rad)
        if remaining < 0.2:
            target_speed = 0.15 + (self.angular_speed - 0.15) * (remaining / 0.2)

        # CẬP NHẬT SỬA LỖI: Hàm set_twist đã được kéo ra ngang hàng với lệnh if
        self.driver.set_twist(0.0, target_speed)

    def stop_test(self):
        self.is_running = False
        self.driver.set_rpm(0,0)
        time.sleep(0.5)
        self.driver.disable_motor()

        self.get_logger().info("=== Kết thúc thử nghiệm ===")
        self.get_logger().info(f"Góc xoay Odom báo cáo: {math.degrees(self.odom_angle_accumulated):.2f} độ.")
        self.get_logger().info("-------------------------------------------------------------")
        self.get_logger().info("HƯỚNG DẪN HIỆU CHỈNH GÓC QUAY:")
        self.get_logger().info("1. Dùng thước đo góc (hoặc nhìn vào vạch đánh dấu trên sàn), kiểm tra xem robot thực tế đã quay được bao nhiêu độ.")
        self.get_logger().info("2. Tính Scale_Factor = Góc_thực_tế / Góc_Odom_báo_cáo")
        self.get_logger().info("3. Cập nhật tham số wheel_base MỚI:")
        self.get_logger().info(f"   wheel_base_mới = {self.wheel_base} / Scale_Factor")
        self.get_logger().info("   (Lưu ý: Với góc xoay, ta CHIA cho Scale_Factor, không phải nhân)")
        self.get_logger().info("-------------------------------------------------------------")

        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = CalibrateAngularNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.driver.set_rpm(0,0)
        node.driver.disable_motor()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
