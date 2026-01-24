#!/usr/bin/env python3
import os
import numpy as np
import rclpy
from rclpy.node import Node # rclpy.node node trong đây là 1 file hay 1 thư viện Node là 1 class là 
from rclpy.time import Time
from rclpy.constants import S_TO_NS # để làm gì 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
from tf2_ros import TransformBroadcaster
def quaternion_from_euler(roll, pitch, yaw):
    # Với roll=pitch=0: x=y=0, z=sin(yaw/2), w=cos(yaw/2)
    return (0.0, 0.0, math.sin(yaw*0.5), math.cos(yaw*0.5))

from geometry_msgs.msg import Twist, TransformStamped        # TB3 dùng Twist, không phải TwistStamped

from sensor_msgs.msg import JointState # trong JointState chứa dữ liệu gì 

from nav_msgs.msg import Odometry # chứa dữ liệu gì 


# def normalize_angle(a): #  tôi thấy nó không xuất hiện trong code bên dưới vậy hàm này có tac dụng gì 
#     # không bắt buộc ở đây, nhưng hữu ích nếu bạn tích hợp tiếp
#     import math
#     a = (a + math.pi) % (2.0 * math.pi) - math.pi
#     return a

class TB3SpeedInspector(Node):
    def __init__(self):
        super().__init__('tb3_speed_inspector')

        # ---- Tham số hình học (đặt đúng theo model) ----
        # Burger: r≈0.033 m, L≈0.160 m
        # Waffle Pi: r≈0.033 m, L≈0.287 m (ước lượng phổ biến)
        default_L = 0.160 if os.environ.get('TURTLEBOT3_MODEL', 'burger') == 'burger' else 0.287
               # L : khoảng cach giua 2 banh
               # r : bán kính bánh xe
        self.declare_parameter('wheel_radius', 0.033)
        self.declare_parameter('wheel_separation', default_L) # truyền tham số 

        self.wheel_radius = float(self.get_parameter('wheel_radius').value)

        self.L = float(self.get_parameter('wheel_separation').value)

        self.get_logger().info(f'Using wheel_radius = {self.wheel_radius:.3f} m')
        self.get_logger().info(f'Using wheel_separation = {self.L:.3f} m')

        # ---- Ma trận chuyển đổi v,ω <-> ωR,ωL ----
        # [v; w] = [[r/2, r/2],[r/L, -r/L]] * [wR; wL]
        self.A = np.array([[self.wheel_radius/2.0, self.wheel_radius/2.0], # chưa hiểu lắm
                           [self.wheel_radius/self.L, -self.wheel_radius/self.L]], dtype=float)

        # ---- Lưu trạng thái trước của JointState để tính sai phân ----
        self.prev_time = None 
        self.prev_pos = {}   # {'left_wheel_joint': pos, 'right_wheel_joint': pos} # cú pháp self.prev_pos = {} {} để trước lúc sau truyền vô

        # ---- QoS sensor data (TB3 phát joint_states với best-effort) ----
        # qos_sensor = QoSProfile(
        #     reliability=ReliabilityPolicy.BEST_EFFORT,
        #     history=HistoryPolicy.KEEP_LAST,
        #     depth=10
        # )
        self.x_ = 0.0
        self.y_ = 0.0
        self.theta_ = 0.0
        self.br_ = TransformBroadcaster(self)   

        

        # ---- Subscriptions ----
        # self.cmd_sub = self.create_subscription(
        #     Twist, '/cmd_vel', self.on_cmd_vel, 10)

        self.js_sub = self.create_subscription(
            JointState, '/joint_states', self.on_joint_states, 10)
        
        self.odom_pub_ = self.create_publisher(Odometry,"odom_danh/odom", 10)
        
        self.odom_msgs_ = Odometry()
        self.odom_msgs_.header.frame_id = "odom"
        self.odom_msgs_.child_frame_id = "base_foot_danh"
        self.odom_msgs_.pose.pose.orientation.x = 0.0
        self.odom_msgs_.pose.pose.orientation.y = 0.0
        self.odom_msgs_.pose.pose.orientation.z = 0.0
        self.odom_msgs_.pose.pose.orientation.w = 1.0

        self.get_logger().info('tb3_speed_inspector is running. Publish some /cmd_vel to see results.')

    # ---------- 1) Đảo kinematics: /cmd_vel -> (ωR, ωL) ----------
    # def on_cmd_vel(self, msg: Twist):
    #     v = msg.linear.x
    #     w = msg.angular.z
    #     robot = np.array([[v], [w]])

    #     # giải A * [wR; wL] = [v; w] giải thích đoạn này
    #     wheel = np.linalg.solve(self.A, robot)   # tốt hơn inv(A) @ robot
    #     wR = float(wheel[0, 0])
    #     wL = float(wheel[1, 0]) # vận tốc gốc bánh trái 

    #     self.get_logger().info(f'cmd_vel -> wheel speeds: wR={wR:.3f} rad/s, wL={wL:.3f} rad/s')

    # ---------- 2) Wheel odometry tức thời: JointState -> (v, ω) ----------
    def on_joint_states(self, msg: JointState):
        # lấy index theo name để khỏi lẫn thứ tự
        try:
            iR = msg.name.index('wheel_right_joint') # cách sửa dụng index nầy là gì
            iL = msg.name.index('wheel_left_joint')
        except ValueError:
            # Một số TB3 dùng 'right_wheel_joint' / 'left_wheel_joint'
            try:
                iR = msg.name.index('right_wheel_joint')
                iL = msg.name.index('left_wheel_joint')
            except ValueError:
                # nếu tên khác nữa, in ra để bạn kiểm tra
                self.get_logger().throttle(2000, f'Unknown joint names: {msg.name}') # throttle là gì cách sửa dụng 
                return
        # iR = msg.name.index('wheel_right_joint')
        # iL = msg.name.index('wheel_left_joint')

        posR = msg.position[iR]
        posL = msg.position[iL] # iL là vị trí của cái name lelf_wheel_joint

        t = Time.from_msg(msg.header.stamp) # giải thích cách lấy thời gian, msg.header.stamp này là gì header stamp là sao
        if self.prev_time is None:
            # khởi tạo mốc
            self.prev_time = t
            self.prev_pos['R'] = posR  # ['R'] là sao 
            self.prev_pos['L'] = posL
            return

        dt = (t - self.prev_time).nanoseconds / S_TO_NS
        if dt <= 0.0 or dt > 0.5:
            # bỏ mẫu bất thường, cập nhật mốc
            self.prev_time = t
            self.prev_pos['R'] = posR
            self.prev_pos['L'] = posL
            return

        dposR = posR - self.prev_pos['R']   # rad delta posion R
        dposL = posL - self.prev_pos['L']   # rad
        wR = dposR / dt                      # rad/s
        wL = dposL / dt                      # rad/s

        # v, w từ công thức chuẩn
        v_linear  = 0.5 * self.wheel_radius * (wR + wL)
        angular = (self.wheel_radius / self.L) * (wR - wL)

        self.prev_time = t
        self.prev_pos['R'] = posR
        self.prev_pos['L'] = posL

        d_s = (self.wheel_radius * dposR + self.wheel_radius * dposL ) /2
        d_theta = (self.wheel_radius * dposR - self.wheel_radius * dposL) /self.L
        self.theta_ += d_theta
        self.x_ += d_s * np.cos(self.theta_)

        self.y_ += d_s * np.sin(self.theta_)


        self.q = quaternion_from_euler(0,0,self.theta_)

        self.odom_msgs_.pose.pose.orientation.x = self.q[0]
        self.odom_msgs_.pose.pose.orientation.y = self.q[1]
        self.odom_msgs_.pose.pose.orientation.z = self.q[2]
        self.odom_msgs_.pose.pose.orientation.w = self.q[3]
        self.odom_msgs_.header.stamp = self.get_clock().now().to_msg() # giống như trên cách nó hoạt động ?
        self.odom_msgs_.pose.pose.position.x = self.x_
        self.odom_msgs_.pose.pose.position.y = self.y_
        self.odom_msgs_.twist.twist.linear.x = v_linear
        self.odom_msgs_.twist.twist.angular.z = angular 



        
        self.transfrom_stamped_ = TransformStamped()
        self.transfrom_stamped_.header.frame_id = "odom"
        self.transfrom_stamped_.child_frame_id = "base_danh"
        self.transfrom_stamped_.transform.translation.x = self.x_
        self.transfrom_stamped_.transform.translation.y = self.y_

        self.transfrom_stamped_.transform.rotation.x = self.q[0]
        self.transfrom_stamped_.transform.rotation.y = self.q[1]
        self.transfrom_stamped_.transform.rotation.z = self.q[2]
        self.transfrom_stamped_.transform.rotation.w = self.q[3]

        self.transfrom_stamped_.header.stamp = self.get_clock().now().to_msg()

        self.br_.sendTransform(self.transfrom_stamped_)



        self.odom_pub_.publish(self.odom_msgs_)
        



        # Log nhẹ nhàng để theo dõi
        #self.get_logger().info(f'joint_states -> v={v:.3f} m/s, w={wz:.3f} rad/s')
        self.get_logger().info(f"x: {self.x_}  y: {self.y_} theta: {self.theta_}")

def main():
    rclpy.init()
    node = TB3SpeedInspector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
