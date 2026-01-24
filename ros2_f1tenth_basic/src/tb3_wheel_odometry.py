#!/usr/bin/env python3
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.constants import S_TO_NS
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy # Không dùng
import math
# from tf2_ros import TransformBroadcaster # <<< ĐÃ XÓA
# from geometry_msgs.msg import TransformStamped # <<< ĐÃ XÓA
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry

def quaternion_from_euler(roll, pitch, yaw):
    return (0.0, 0.0, math.sin(yaw*0.5), math.cos(yaw*0.5))

class TB3WheelOdometry(Node): # Đổi tên class cho rõ ràng
    def __init__(self):
        super().__init__('tb3_wheel_odometry')

        # ---- Tham số hình học ----
        default_L = 0.160 if os.environ.get('TURTLEBOT3_MODEL', 'burger') == 'burger' else 0.287
        self.declare_parameter('wheel_radius', 0.033)
        self.declare_parameter('wheel_separation', default_L)

        self.wheel_radius = float(self.get_parameter('wheel_radius').value)
        self.L = float(self.get_parameter('wheel_separation').value)

        self.get_logger().info(f'Using wheel_radius = {self.wheel_radius:.3f} m')
        self.get_logger().info(f'Using wheel_separation = {self.L:.3f} m')

        # ---- Lưu trạng thái trước ----
        self.prev_time = None
        self.prev_pos = {}

        self.x_ = 0.0
        self.y_ = 0.0
        self.theta_ = 0.0
        
        # self.br_ = TransformBroadcaster(self) # <<< ĐÃ XÓA

        # ---- Subscriptions ----
        self.js_sub = self.create_subscription(
            JointState, '/joint_states', self.on_joint_states, 10)
        
        # ---- Publisher ----
        self.odom_pub_ = self.create_publisher(Odometry, "odom_danh/odom", 10)
        
        self.odom_msgs_ = Odometry()
        self.odom_msgs_.header.frame_id = "odom"
        
        # <<< THAY ĐỔI QUAN TRỌNG: Dùng frame tiêu chuẩn
        self.odom_msgs_.child_frame_id = "base_footprint" 
        
        self.odom_msgs_.pose.pose.orientation.w = 1.0

        self.get_logger().info('tb3_wheel_odometry is running...')

    def on_joint_states(self, msg: JointState):
        try:
            iR = msg.name.index('wheel_right_joint')
            iL = msg.name.index('wheel_left_joint')
        except ValueError:
            try:
                iR = msg.name.index('right_wheel_joint')
                iL = msg.name.index('left_wheel_joint')
            except ValueError:
                self.get_logger().throttle(2000, f'Unknown joint names: {msg.name}')
                return

        posR = msg.position[iR]
        posL = msg.position[iL]

        t = Time.from_msg(msg.header.stamp)
        if self.prev_time is None:
            self.prev_time = t
            self.prev_pos['R'] = posR
            self.prev_pos['L'] = posL
            return

        dt = (t - self.prev_time).nanoseconds / S_TO_NS
        if dt <= 0.0 or dt > 0.5:
            self.prev_time = t
            self.prev_pos['R'] = posR
            self.prev_pos['L'] = posL
            return

        dposR = posR - self.prev_pos['R']
        dposL = posL - self.prev_pos['L']
        wR = dposR / dt
        wL = dposL / dt

        v_linear  = 0.5 * self.wheel_radius * (wR + wL)
        v_angular = (self.wheel_radius / self.L) * (wR - wL) # đổi tên biến

        self.prev_time = t
        self.prev_pos['R'] = posR
        self.prev_pos['L'] = posL

        d_s = (self.wheel_radius * dposR + self.wheel_radius * dposL ) / 2.0
        d_theta = (self.wheel_radius * dposR - self.wheel_radius * dposL) / self.L
        
        self.theta_ += d_theta
        self.x_ += d_s * np.cos(self.theta_ + d_theta / 2.0) # Cải tiến nhỏ (Runge-Kutta 2)
        self.y_ += d_s * np.sin(self.theta_ + d_theta / 2.0)

        self.q = quaternion_from_euler(0, 0, self.theta_)

        # ---- Điền vào message Odometry ----
        self.odom_msgs_.header.stamp = self.get_clock().now().to_msg()
        self.odom_msgs_.pose.pose.position.x = self.x_
        self.odom_msgs_.pose.pose.position.y = self.y_
        self.odom_msgs_.pose.pose.orientation.x = self.q[0]
        self.odom_msgs_.pose.pose.orientation.y = self.q[1]
        self.odom_msgs_.pose.pose.orientation.z = self.q[2]
        self.odom_msgs_.pose.pose.orientation.w = self.q[3]
        
        self.odom_msgs_.twist.twist.linear.x = v_linear
        self.odom_msgs_.twist.twist.angular.z = v_angular
        
        self.odom_pub_.publish(self.odom_msgs_)

        # ---- TOÀN BỘ KHỐI PUBLISH TF ĐÃ BỊ XÓA ----
        # self.transfrom_stamped_ = TransformStamped()
        # ...
        # self.br_.sendTransform(self.transfrom_stamped_)
        
        self.get_logger().info(f"Wheel Odom: x={self.x_:.2f}, y={self.y_:.2f}, th={self.theta_:.2f}")

def main():
    rclpy.init()
    node = TB3WheelOdometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
