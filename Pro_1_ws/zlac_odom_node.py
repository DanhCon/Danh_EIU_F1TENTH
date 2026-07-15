#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
import tf2_ros
import math
import time
import numpy as np

from zlac8015d_driver import ZLAC8015D_Driver

class ZLAC8015DOdomNode(Node):
    def __init__(self):
        super().__init__('zlac_odom_node')

        self.reent_group = ReentrantCallbackGroup()

        # Thong so chuan cua EIU-FABLAB-AMR
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('wheel_radius', 0.0535)
        self.declare_parameter('wheel_base', 0.45)
        self.declare_parameter('cpr', 4096)
        self.declare_parameter('travel_in_1_vong', 0.336)
        self.declare_parameter('max_rpm', 150)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('cmd_vel_timeout', 0.1) 

        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value
        self.wheel_radius = self.get_parameter('wheel_radius').get_parameter_value().double_value
        self.wheel_base = self.get_parameter('wheel_base').get_parameter_value().double_value
        self.cpr = self.get_parameter('cpr').get_parameter_value().integer_value
        self.travel_in_1_vong = self.get_parameter('travel_in_1_vong').get_parameter_value().double_value
        self.max_rpm = self.get_parameter('max_rpm').get_parameter_value().integer_value
        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').get_parameter_value().double_value

        # Gioi han toc do EIU
        self.max_linear_velocity = 1.6
        self.max_angular_velocity = 0.8

        # Bộ lọc làm mượt tốc độ (Velocity Smoother)
        self.declare_parameter('linear_accel', 0.6)   # m/s^2
        self.declare_parameter('angular_accel', 0.8)  # rad/s^2
        self.linear_accel = self.get_parameter('linear_accel').value
        self.angular_accel = self.get_parameter('angular_accel').value
        
        self.target_linear = 0.0
        self.target_angular = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0

        self.get_logger().info(f'Dang ket noi driver tai cong {self.port}')

        self.driver = ZLAC8015D_Driver(
            port=self.port, baudrate=self.baudrate,
            wheel_radius=self.wheel_radius, wheel_base=self.wheel_base,
            cpr=self.cpr, travel_in_1_vong=self.travel_in_1_vong,
            max_rpm=self.max_rpm
        )
        
        if not self.driver.init_motor():
            self.get_logger().info(f'Khong the ket noi hoac khoi tao dong co')
        else:
            self.get_logger().info("Da ket noi va kich hoat dong co thanh cong nha ::>")
            # ÉP ĐỒNG BỘ GIA TỐC PHẦN CỨNG (Tránh việc 2 bánh khởi động lệch pha)
            try:
                self.driver.set_accel_time(200, 200)
                self.driver.set_decel_time(200, 200)
            except Exception as e:
                pass

        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) 
        
        # SUBSCRIBER (GHI Modbus): Da luong
        self.cmd_vel_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10, callback_group=self.reent_group)
        
        self.reset_odom_srv = self.create_service(Trigger, 'reset_odom', self.reset_odom_callback, callback_group=self.reent_group)

        self.last_time = self.get_clock().now()
        self.last_cmd_vel_time = self.get_clock().now()

        timer_period = 1.0 / self.publish_rate
        # TIMER (DOC Modbus): Da luong
        self.timer = self.create_timer(timer_period, self.timer_callback, callback_group=self.reent_group)

        self.is_shutting_down = False
        self.get_logger().info('Node Odom hoat dong DA LUONG (MultiThread) da san sang ')

    def cmd_vel_callback(self, msg:Twist):
        """ Nhan lenh toc do (Twist) va luu vao target, KHONG gui xuong motor ngay lap tuc de lam muot"""
        if self.is_shutting_down:
            return
        self.last_cmd_vel_time = self.get_clock().now()
        
        # 1. Gioi han toc do dau vao nhu EIU + Kiem tra DEADZONE (Tranh tay cam bi troi)
        joy_linear = msg.linear.x
        joy_angular = msg.angular.z
        
        # Deadzone 0.01 m/s hoac rad/s (Tranh tay cam bi troi nhung khong bop chet do nhay)
        if abs(joy_linear) < 0.01:
            joy_linear = 0.0
        if abs(joy_angular) < 0.01: 
            joy_angular = 0.0
        
        self.target_linear = np.clip(joy_linear, -self.max_linear_velocity, self.max_linear_velocity)
        self.target_angular = np.clip(joy_angular, -self.max_angular_velocity, self.max_angular_velocity)

    def reset_odom_callback(self, req, res):
        """ Callback: Reset toa do odom khi nhan dc tin reset"""
        self.get_logger().info('dang thuc hien reset Odom .....')
        self.driver.reset_odom()
        res.success = True
        res.message = "Da reset toa do Odometry ve 0"
        return res

    def timer_callback(self):
        if self.is_shutting_down: 
            return
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        # 1. Kiem tra Timeout, neu lau qua khong nhan duoc lenh tu Tay Cam -> Cho xe dung lai
        time_since_last_cmd = (now - self.last_cmd_vel_time).nanoseconds / 1e9
        if time_since_last_cmd > self.cmd_vel_timeout:
            self.target_linear = 0.0
            self.target_angular = 0.0
            
        # 2. XU LY LAM MUOT TOC DO (VELOCITY SMOOTHER)
        # Tang/Giam toc do hien tai (current) huong ve toc do muc tieu (target) tu tu
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
        
        # Chi gui lenh xuong RS485 neu toc do dang BIEN THIEN hoac Chua Dung Im (tranh span RS485)
        if (new_linear != self.current_linear) or (new_angular != self.current_angular) or (self.current_linear != 0.0) or (self.current_angular != 0.0):
            self.current_linear = new_linear
            self.current_angular = new_angular
            
            # Tinh Inverse Kinematics
            right_wheel = self.current_linear + (self.current_angular * self.wheel_base / 2.0)
            left_wheel = self.current_linear - (self.current_angular * self.wheel_base / 2.0)
            left_rpm = (left_wheel / self.wheel_radius) * 60.0 / (2 * math.pi)
            right_rpm = (right_wheel / self.wheel_radius) * 60.0 / (2 * math.pi)
            
            if self.driver.connected:
                try:
                    self.get_logger().info(f"Target:[L={self.target_linear:.2f}, A={self.target_angular:.2f}] | RPM Gửi Đi: [Trái={int(left_rpm)}, Phải={int(right_rpm)}]")
                    self.driver.set_rpm(int(left_rpm), int(right_rpm))
                except Exception as e:
                    self.get_logger().error(f"Loi set_rpm: {e}")
                
        # 3. DOC Modbus & Tinh Odom
        x, y, theta, v_x, w_z = self.driver.update_odometry(dt)

        cy = math.cos(theta*0.5)
        sy = math.sin(theta*0.5)
        qx, qy, qz, qw = 0.0, 0.0, sy, cy

        odom_msg = Odometry()
        odom_msg.header.stamp = now.to_msg()
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame

        odom_msg.pose.pose.position.x = x 
        odom_msg.pose.pose.position.y = y # SỬA LỖI (trước đây là y = x)
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
        self.get_logger().info('dang tat node, dung dong co')
        if self.driver.connected:
            self.driver.set_rpm(0,0)
            time.sleep(0.1)
            self.driver.disable_motor()
            self.driver.close_connect()

def main(args = None):
    rclpy.init(args = args)
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

if __name__  == '__main__':
    main()
