import rclpy
from rclpy.node import Node
import time
import math
from zlac8015d_driver import ZLAC8015D_Driver

class CalibrateLinearNode(Node):
    def __init__(self):
        super().__init__('calibrate_linear_node')

        self.declare_parameter('port' , '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('wheel_radius', 0.0535)
        self.declare_parameter('wheel_base',0.45)
        self.declare_parameter('cpr', 4096)
        self.declare_parameter('travel_in_1_vong', 0.3830)
        self.declare_parameter('test_distance', 1.0)

        self.declare_parameter('speed', 0.2)

        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value
        self.wheel_radius = self.get_parameter('wheel_radius').get_parameter_value().double_value
        self.wheel_base = self.get_parameter('wheel_base').get_parameter_value().double_value
        self.cpr = self.get_parameter('cpr').get_parameter_value().integer_value
        self.travel_in_one_rev = self.get_parameter('travel_in_1_vong').get_parameter_value().double_value
        self.test_distance = self.get_parameter('test_distance').get_parameter_value().double_value
        self.speed = self.get_parameter('speed').get_parameter_value().double_value

        self.get_logger().info(f"===bat dau ")

        self.get_logger().info(f"muc tieu: chay thang {self.test_distance} met voi toc do {self.speed} m/s")

        self.driver = ZLAC8015D_Driver(
            port = self.port,
            baudrate= self.baudrate,
            wheel_base= self.wheel_base,
            wheel_radius= self.wheel_radius,
            cpr = self.cpr,
            travel_in_1_vong= self.travel_in_one_rev,
        )
        if not self.driver.init_motor():
            self.get_logger().error(' Loi ket noi dong co ')
            return 
        self.odom_distance = 0.0
        self.is_running = True
        self.counter =0

        self.driver.update_odometry(0.0)
        self.last_time = time.time()

        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):

        if not self.is_running:
            return 
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if self.counter == 0:
            self.driver.set_twist(0.0, 0.0)
            self.counter = 1


        x,y,theta, v_x, w_z = self.driver.update_odometry(dt)

        self.odom_distance = math.sqrt(x*x + y*y)

        print(f"\rXe da di chuyen {self.odom_distance:.4f} m / {self.test_distance} m   ", end="", flush=True)

        remaining = abs(self.test_distance) - self.odom_distance

        if remaining <= 0:
            print()
            self.stop_test()

            return 
        target_speed = self.speed
        if remaining < 0.15:
            target_speed = 0.02 + (self.speed-0.02)*(remaining/0.15)

        self.driver.set_twist(target_speed, 0.0)

    def stop_test(self):
        self.is_running = False
        self.driver.set_rpm(0,0)
        time.sleep(0.5)
        

        self.get_logger().info("=== Ket thuc thu nghiem===")
        self.get_logger().info(f"Quãng đường Odom báo cáo: {self.odom_distance:.4f} mét.")
        self.get_logger().info("-------------------------------------------------------------")
        self.get_logger().info("HƯỚNG DẪN HIỆU CHỈNH:")
        self.get_logger().info("1. Dùng thước mét đo quãng đường đi thực tế của robot trên sàn (D_thực_tế).")
        self.get_logger().info(f"2. Tính Scale_Factor = D_thực_tế / {self.odom_distance:.4f}")
        self.get_logger().info(f"3. Cập nhật tham số mới:")
        self.get_logger().info(f"   wheel_radius_mới = {self.wheel_radius} * Scale_Factor")
        self.get_logger().info(f"   travel_in_1_vong_mới = {self.travel_in_one_rev} * Scale_Factor")
        self.get_logger().info("-------------------------------------------------------------")

        self.destroy_node()
        rclpy.shutdown()

def main(args = None):
    rclpy.init(args= args)
    node = CalibrateLinearNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.driver.set_rpm(0,0)
        node.driver.disable_motor()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
