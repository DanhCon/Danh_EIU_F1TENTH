#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import csv
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.duration import Duration
from copy import deepcopy
import time
# --- IMPORT VISUALIZATION ---
from visualization_msgs.msg import Marker, MarkerArray
# ----------------------------

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs
from rclpy.duration import Duration
from geometry_msgs.msg import PointStamped

class test_transfrom(Node):
    def __init__(self):
        super().__init__("test_transfrom")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer , self)
        self.car_frame = "base_link"
        self.map_frame = "map"

        # Pub/Sub
        self.sub_odom = self.create_subscription(Odometry , "odom", self.odom_callback, 10)
        self.pub_marker_1 = self.create_publisher(Marker, "/vi_tri_odom", 10)
        self.pub_marker_2 = self.create_publisher(Marker, "/vi_tri_da_tranform", 10)
    def odom_callback(self, msg: Odometry):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, 
                self.car_frame, 
                rclpy.time.Time(seconds=0),
                Duration(seconds=0.05)
            )
            robot_x_map = transform.transform.translation.x
            robot_y_map = transform.transform.translation.y
            x_pos = msg.pose.pose.position.x
            y_pos = msg.pose.pose.position.y

            self.publish_vi_tri_da_trans(robot_x_map, robot_y_map)
            self.publish_bth(x_pos, y_pos)
            
        except TransformException:
            return
    def publish_vi_tri_da_trans(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "publish_vi_tri_da_trans"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3
        marker.color.a = 1.0; marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        self.pub_marker_2.publish(marker)
    def publish_bth(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "publish_vi_tri_bth"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3
        marker.color.a = 1.0; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        self.pub_marker_2.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = test_transfrom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    except Exception as e: print(f"Error: {e}")
    finally:
        if rclpy.ok():
            try:
                stop_msg = AckermannDriveStamped()
                node.pub_drive.publish(stop_msg)
                time.sleep(0.1)
            except: pass
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()        

    