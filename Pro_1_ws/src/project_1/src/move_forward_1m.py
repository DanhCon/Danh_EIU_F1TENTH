#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class MoveForwardNode(Node):
    def __init__(self):
        super().__init__('move_forward_node')

        self.declare_parameter('distance', 1.0)          # Khoảng cách tiến (m)
        self.declare_parameter('max_linear_vel', 0.25)   # Tốc độ tối đa (m/s)
        self.declare_parameter('min_linear_vel', 0.05)   # Tốc độ tối thiểu (m/s)
        self.declare_parameter('kp', 1.0)                # Hệ số P

        self.target_dist = self.get_parameter('distance').value
        self.max_v = self.get_parameter('max_linear_vel').value
        self.min_v = self.get_parameter('min_linear_vel').value
        self.kp = self.get_parameter('kp').value

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.is_running = True
        self.odom_received = False
        self.start_x = None
        self.start_y = None
        self.current_x = 0.0
        self.current_y = 0.0

        self.get_logger().info(f"=== BẮT ĐẦU TIẾN THẲNG {self.target_dist:.2f}M ===")

    def odom_callback(self, msg: Odometry):
        self.odom_received = True
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def control_loop(self):
        if not self.is_running or not self.odom_received:
            return

        if self.start_x is None:
            self.start_x = self.current_x
            self.start_y = self.current_y
            return

        moved_dist = math.hypot(self.current_x - self.start_x, self.current_y - self.start_y)
        rem_dist = self.target_dist - moved_dist

        cmd = Twist()

        # Dừng khi đạt khoảng cách (sai số <= 1.5cm)
        if rem_dist <= 0.015:
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            self.is_running = False
            self.get_logger().info(f"✅ ĐÃ HOÀN THÀNH TIẾN THẲNG! (Quãng đường: {moved_dist:.3f}m)")
            return

        v = self.kp * rem_dist
        v = max(self.min_v, min(self.max_v, v))
        cmd.linear.x = v
        self.cmd_pub.publish(cmd)

        self.get_logger().info(f"Đang tiến: {moved_dist:.2f}m / {self.target_dist:.2f}m", throttle_duration_sec=0.5)

def main(args=None):
    rclpy.init(args=args)
    node = MoveForwardNode()
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