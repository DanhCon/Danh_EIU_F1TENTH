#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
import tf2_ros
import math
import time
import numpy as np
import threading

from zlac8015d_driver import ZLAC8015D_Driver

class ZLAC8015DOdomNode(Node):
    def __init__(self):
        super().__init__('zlac_odom_node')

        # Sử dụng MutuallyExclusiveCallbackGroup để bảo vệ giao tiếp Serial RS485 không bị xung đột đa luồng
        self.modbus_group = MutuallyExclusiveCallbackGroup()
        self.driver_lock = threading.Lock()

        # Khai báo parameters
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('wheel_radius', 0.0535)
        self.declare_parameter('wheel_base', 0.45)
        self.declare_parameter('cpr', 4096)

        self.declare_parameter('max_rpm', 150)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('cmd_vel_timeout', 0.2)

        # Giới hạn tốc độ robot (Parameter hóa thay vì hardcode)
        self.declare_parameter('max_linear_velocity', 1.6)
        self.declare_parameter('max_angular_velocity', 0.8)

        # Bộ lọc làm mượt tốc độ (Velocity Smoother)
        self.declare_parameter('linear_accel', 0.6)   # m/s^2
        self.declare_parameter('angular_accel', 0.8)  # rad/s^2

        # Đọc tham số sử dụng thuộc tính .value đơn giản
        self.port = self.get_parameter('port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.cpr = self.get_parameter('cpr').value

        self.max_rpm = self.get_parameter('max_rpm').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').value

        self.max_linear_velocity = self.get_parameter('max_linear_velocity').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        self.linear_accel = self.get_parameter('linear_accel').value
        self.angular_accel = self.get_parameter('angular_accel').value

        self.target_linear = 0.0
        self.target_angular = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0

        self.get_logger().info(f'Đang kết nối driver tại cổng {self.port}...')

        self.driver = ZLAC8015D_Driver(
            port=self.port, baudrate=self.baudrate,
            wheel_radius=self.wheel_radius, wheel_base=self.wheel_base,
            cpr=self.cpr, 
            max_rpm=self.max_rpm
        )
        
        if not self.driver.init_motor():
            self.get_logger().error(f'Không thể kết nối hoặc khởi tạo động cơ!')
        else:
            self.get_logger().info("Đã kết nối và kích hoạt động cơ thành công.")
            try:
                with self.driver_lock:
                    self.driver.set_accel_time(200, 200)
                    self.driver.set_decel_time(200, 200)
            except Exception as e:
                self.get_logger().warn(f"Không thể đặt gia tốc phần cứng: {e}")

        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) 
        
        # SUBSCRIBER
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10
        )
        
        # SERVICE (Dùng modbus_group để tránh xung đột Modbus)
        self.reset_odom_srv = self.create_service(
            Trigger, 'reset_odom', self.reset_odom_callback, callback_group=self.modbus_group
        )

        self.last_time = self.get_clock().now()
        self.last_cmd_vel_time = self.get_clock().now()

        # TIMER (Dùng modbus_group)
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(
            timer_period, self.timer_callback, callback_group=self.modbus_group
        )

        self.is_shutting_down = False
        self.get_logger().info('Node ZLAC8015D Odom đã sẵn sàng hoạt động.')

    def cmd_vel_callback(self, msg: Twist):
        if self.is_shutting_down:
            return
        self.last_cmd_vel_time = self.get_clock().now()
        
        joy_linear = msg.linear.x
        joy_angular = msg.angular.z
        
        # Deadzone 0.01 m/s hoặc rad/s
        if abs(joy_linear) < 0.01:
            joy_linear = 0.0
        if abs(joy_angular) < 0.01: 
            joy_angular = 0.0
        
        self.target_linear = float(np.clip(joy_linear, -self.max_linear_velocity, self.max_linear_velocity))
        self.target_angular = float(np.clip(joy_angular, -self.max_angular_velocity, self.max_angular_velocity))

    def reset_odom_callback(self, req, res):
        self.get_logger().info('Đang thực hiện reset Odom...')
        with self.driver_lock:
            if self.driver.connected:
                self.driver.reset_odom()
        res.success = True
        res.message = "Đã reset tọa độ Odometry về 0"
        return res

    def timer_callback(self):
        if self.is_shutting_down: 
            return
            
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        # Bảo vệ tránh dt bằng 0 hoặc âm khi khởi chạy/lag
        if dt <= 0.0:
            return

        # 1. Kiểm tra Timeout dừng xe khi mất tín hiệu cmd_vel
        time_since_last_cmd = (now - self.last_cmd_vel_time).nanoseconds / 1e9
        if time_since_last_cmd > self.cmd_vel_timeout:
            self.target_linear = 0.0
            self.target_angular = 0.0
            
        # 2. XỬ LÝ LÀM MƯỢT TỐC ĐỘ (VELOCITY SMOOTHER)
        step_lin = self.linear_accel * dt
        step_ang = self.angular_accel * dt
        
        def move_towards(current, target, max_delta):
            if current < target:
                return min(current + max_delta, target)
            elif current > target:
                return max(current - max_delta, target)
            return current
            
        new_linear = move_towards(self.current_linear, self.target_linear, step_lin)
        new_angular = move_towards(self.current_angular, self.target_angular, step_ang)
        
        # Chỉ gửi lệnh xuống RS485 nếu tốc độ đang thay đổi hoặc khác 0
        if (new_linear != self.current_linear) or (new_angular != self.current_angular) or (self.current_linear != 0.0) or (self.current_angular != 0.0):
            self.current_linear = new_linear
            self.current_angular = new_angular
            
            # Tính Kinematics bánh xe
            right_wheel = self.current_linear + (self.current_angular * self.wheel_base / 2.0)
            left_wheel = self.current_linear - (self.current_angular * self.wheel_base / 2.0)
            left_rpm = (left_wheel / self.wheel_radius) * 60.0 / (2 * math.pi)
            right_rpm = (right_wheel / self.wheel_radius) * 60.0 / (2 * math.pi)
            
            with self.driver_lock:
                if self.driver.connected:
                    try:
                        # Giới hạn tần suất log (chỉ in 1 giây 1 lần để tránh spam console)
                        self.get_logger().info(
                            f"Target:[L={self.target_linear:.2f}, A={self.target_angular:.2f}] | RPM: [L={int(left_rpm)}, R={int(right_rpm)}]",
                            throttle_duration_sec=1.0
                        )
                        self.driver.set_rpm(int(left_rpm), int(right_rpm))
                    except Exception as e:
                        self.get_logger().error(f"Lỗi set_rpm RS485: {e}")

        # 3. ĐỌC Modbus & TÍNH ODOMETRY (Bọc try-except an toàn)
        try:
            with self.driver_lock:
                odom_data = self.driver.update_odometry(dt)
            if odom_data is None:
                return
            x, y, theta, v_x, w_z = odom_data
        except Exception as e:
            self.get_logger().error(f"Lỗi khi đọc Odometry qua RS485: {e}", throttle_duration_sec=2.0)
            return

        # 4. PUBLISH ODOMETRY & TF
        cy = math.cos(theta * 0.5)
        sy = math.sin(theta * 0.5)
        qx, qy, qz, qw = 0.0, 0.0, sy, cy

        odom_msg = Odometry()
        odom_msg.header.stamp = now.to_msg()
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame

        odom_msg.pose.pose.position.x = x 
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw

        odom_msg.twist.twist.linear.x = v_x
        odom_msg.twist.twist.angular.z = w_z

        self.odom_pub.publish(odom_msg)

        tf = TransformStamped()
        tf.header.stamp = now.to_msg()
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame

        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(tf)

    def stop(self):
        self.is_shutting_down = True
        self.get_logger().info('Đang tắt Node, ngắt kết nối động cơ...')
        if hasattr(self, 'timer'):
            self.timer.cancel()
            
        with self.driver_lock:
            if self.driver.connected:
                try:
                    self.driver.set_rpm(0, 0)
                    time.sleep(0.1)
                    self.driver.disable_motor()
                    self.driver.close_connect()
                except Exception as e:
                    self.get_logger().warn(f"Lỗi khi ngắt kết nối động cơ: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ZLAC8015DOdomNode()
    
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()