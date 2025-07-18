#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import time
import sympy as sp

class Move_robot(Node):
    def __init__(self):
        from rclpy.node import Node, NodeOptions

        options = NodeOptions()
        options.automatically_declare_parameters_from_overrides = True
        super().__init__('publisher_node', options=options)

        print("publisher_node initialized ")
        self.get_logger().info("publisher_node initialized")

        # ✅ Khai báo tham số
        # ✅ Khai báo và set giá trị mặc định để ROS2 nhận
        self.declare_parameter("stop_distance", 2.0)
        self.set_parameters([
            rclpy.parameter.Parameter('stop_distance', rclpy.Parameter.Type.DOUBLE, 2.0)
        ])


        
        self.timer = self.create_timer(1, self.timer_callback)
        self.publisher_ = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        self._sub = self.create_subscription(Odometry, '/odom', self.listen, 10)
        self.initialize = None
        self.currentPose = None
        self.cmd = AckermannDriveStamped()

    def timer_callback(self):
        if self.initialize is None or self.currentPose is None:
            return

        # ✅ Lấy giá trị tham số
        stop_distance = self.get_parameter("stop_distance").value

        distance = sp.sqrt(
            (self.currentPose[0] - self.initialize[0]) ** 2 +
            (self.currentPose[1] - self.initialize[1]) ** 2
        )

        if distance > stop_distance:
            self.cmd.drive.speed = 0.0
        else:
            self.cmd.drive.speed = 0.5
            self.cmd.drive.steering_angle = 0.0

        self.publisher_.publish(self.cmd)

    def listen(self, msg: Odometry):
        self.currentPose = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        print(self.currentPose)
        if self.initialize is None:
            self.initialize = self.currentPose


def main():
    rclpy.init()
    node = Move_robot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
