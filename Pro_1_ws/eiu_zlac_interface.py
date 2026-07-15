#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from zlac8015d_driver import ZLAC8015D_Driver
import time
import math

class EIU_ZlacInterface(Node):
    def __init__(self):
        super().__init__('eiu_zlac_interface_node')
        
        # Nhóm callback cho phép đa luồng
        self.reent_group = ReentrantCallbackGroup()
        
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        
        port = self.get_parameter('port').get_parameter_value().string_value
        baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value
        
        # Kế thừa thông số gốc
        self.driver = ZLAC8015D_Driver(
            port=port, baudrate=baudrate,
            wheel_radius=0.0535, wheel_base=0.45,
            cpr=4096, travel_in_1_vong=0.336
        )
        
        if not self.driver.init_motor():
            self.get_logger().error("Khong the ket noi ZLAC8015D!")
            return
            
        self.get_logger().info("Khoi tao dong co (Kem gia toc 200ms) THANH CONG!")

        # SUBSCRIBER (GHI Modbus): Nhận RPM từ Diff Controller
        self.vel_sub = self.create_subscription(
            Float64MultiArray, 
            'eiu/wheel_rotational_vel', 
            self.sub_vel_callback, 
            10, 
            callback_group=self.reent_group
        )
        
        # PUBLISHER (DOC Modbus): Timer Doc Odom
        self.joint_pub = self.create_publisher(JointState, 'eiu/joint_states', 10)


        self.timer = self.create_timer(0.1, self.pub_joint_callback, callback_group=self.reent_group)
        
        self.is_shutting_down = False

    def sub_vel_callback(self, msg):
        if self.is_shutting_down: return
        if len(msg.data) != 2: return
        
        # EIU format: [left_rpm, right_rpm]
        left_rpm = msg.data[0]
        right_rpm = msg.data[1]
        
        try:
            self.driver.set_rpm(int(left_rpm), int(right_rpm))
        except Exception as e:
            self.get_logger().error(f"Loi set_rpm: {e}")

    def pub_joint_callback(self):
        if self.is_shutting_down: return
        
        l_travel, r_travel = self.driver.get_wheels_travelled()
        if l_travel is None or r_travel is None:
            return
            
        # Tương tự như EIU Odom, gửi quãng đường/vị trí qua JointState
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = ['left_wheel', 'right_wheel']
        # Đẩy trực tiếp quãng đường đi được (mét) vào position
        js.position = [l_travel, r_travel] 
        self.joint_pub.publish(js)
        
    def stop(self):
        self.is_shutting_down = True
        self.driver.set_rpm(0, 0)
        time.sleep(0.1)
        self.driver.disable_motor()

def main():
    rclpy.init()
    node = EIU_ZlacInterface()
    
    # EIU sử dụng MultiThreadedExecutor!
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Dong Interface Node...")
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
