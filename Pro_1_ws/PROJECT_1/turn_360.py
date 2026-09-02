#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class Turn360Node(Node):
    def __init__(self):
        super().__init__('turn_360_node')

        self.declare_parameter('target_deg', 360.0)     # Quay 360 độ
        self.declare_parameter('max_angular_vel', 0.4)
        self.declare_parameter('min_angular_vel', 0.12)
        self.declare_parameter('kp', 1.2)

        self.target_rad = math.radians(self.get_parameter('target_deg').value)
        self.max_w = self.get_parameter('max_angular_vel').value
        self.min_w = self.get_parameter('min_angular_vel').value
        self.kp = self.get_parameter('kp').value

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.is_running = True
        self.odom_received = False
        self.last_yaw = None
        self.accumulated_yaw = 0.0

        self.get_logger().info("=== BẮT ĐẦU QUAY 360 ĐỘ ===")

    def get_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle_diff(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def odom_callback(self, msg: Odometry):
        self.odom_received = True
        current_yaw = self.get_yaw(msg.pose.pose.orientation)

        if self.is_running:
            if self.last_yaw is not None:
                d_yaw = self.normalize_angle_diff(current_yaw - self.last_yaw)
                self.accumulated_yaw += d_yaw
            self.last_yaw = current_yaw

    def control_loop(self):
        if not self.is_running or not self.odom_received:
            return

        rem_rad = abs(self.target_rad) - abs(self.accumulated_yaw)
        current_deg = math.degrees(abs(self.accumulated_yaw))
        cmd = Twist()

        if rem_rad <= math.radians(0.5):
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.is_running = False
            self.get_logger().info(f"✅ ĐÃ HOÀN THÀNH QUAY 360°! (Góc đạt: {current_deg:.2f}°)")
            return

        w = self.kp * rem_rad
        w = max(self.min_w, min(self.max_w, w))
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        self.get_logger().info(f"Đang quay 360: {current_deg:.1f}° / 360.0°", throttle_duration_sec=0.5)

def main(args=None):
    rclpy.init(args=args)
    node = Turn360Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        node.cmd_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()