#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
# from rosbasic_msgs.srv import GetTransform
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster, TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from transformations import quaternion_from_euler, quaternion_multiply, quaternion_inverse



class SimpleTfKinematics(Node):

    def __init__(self):
        super().__init__("simple_tf_kinematics")
        # self.x_increment_ = 1.0
        # self.last_x_ = 0.0
        # self.rotations_counter_ = 0
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.timerCallback, 10)

        # TF Broadcaster
        self.static_tf_broadcaster_ = StaticTransformBroadcaster(self)
        self.dynamic_tf_broadcaster_ = TransformBroadcaster(self)
        self.static_transform_stamped_ = TransformStamped()

        self.static_transform_imu = TransformStamped()
        self.static_transform_laser = TransformStamped()


        self.dynamic_transform_stamped_ = TransformStamped()

        # TF Listener
        self.tf_buffer_ = Buffer()
        self.tf_listener_ = TransformListener(self.tf_buffer_, self)

        self.static_transform_stamped_.header.stamp = self.get_clock().now().to_msg()
        self.static_transform_stamped_.header.frame_id = "base_link"
        self.static_transform_stamped_.child_frame_id = "camera_link"
        self.static_transform_stamped_.transform.translation.x = 5.0
        self.static_transform_stamped_.transform.translation.y = 0.0
        self.static_transform_stamped_.transform.translation.z = 0.0
        self.static_transform_stamped_.transform.rotation.x = 0.0
        self.static_transform_stamped_.transform.rotation.y = 0.0
        self.static_transform_stamped_.transform.rotation.z = 0.0
        self.static_transform_stamped_.transform.rotation.w = 1.0





        self.static_tf_broadcaster_.sendTransform([self.static_transform_stamped_,self.static_transform_laser,self.static_transform_imu])
        
        self.last_orientation_ = quaternion_from_euler(0, 0, 0)

        self.orientation_increment_ = quaternion_from_euler(0, 0, 0.05)
    

    def timerCallback(self,msg:Odometry):
        self.dynamic_transform_stamped_.header.stamp = self.get_clock().now().to_msg()
        self.dynamic_transform_stamped_.header.frame_id = "odom_danh"
        self.dynamic_transform_stamped_.child_frame_id = "base_link_danh"
        self.dynamic_transform_stamped_.transform.translation.x = msg.pose.pose.position.x
        self.dynamic_transform_stamped_.transform.translation.y = msg.pose.pose.position.y
        self.dynamic_transform_stamped_.transform.translation.z = msg.pose.pose.position.z

        self.dynamic_transform_stamped_.transform.rotation.x = msg.pose.pose.orientation.x
        self.dynamic_transform_stamped_.transform.rotation.y = msg.pose.pose.orientation.y
        self.dynamic_transform_stamped_.transform.rotation.z = msg.pose.pose.orientation.z
        self.dynamic_transform_stamped_.transform.rotation.w = msg.pose.pose.orientation.w

        # Euler to Quaternion
        # q = quaternion_multiply(self.last_orientation_, self.orientation_increment_)

        # self.dynamic_transform_stamped_.transform.rotation.x = q[0]
        # self.dynamic_transform_stamped_.transform.rotation.y = q[1]
        # self.dynamic_transform_stamped_.transform.rotation.z = q[2]
        # self.dynamic_transform_stamped_.transform.rotation.w = q[3]

        self.dynamic_tf_broadcaster_.sendTransform(self.dynamic_transform_stamped_)
        
        # self.last_x_ = self.dynamic_transform_stamped_.transform.translation.x
        # self.last_orientation_ = q
        # # self.rotations_counter_ += 1
        # if self.rotations_counter_ >= 100:
        #     self.orientation_increment_ = quaternion_inverse(self.orientation_increment_)
        #     self.rotations_counter_ = 0

    
    def getTransformCallback(self, req, res):
        self.get_logger().info("Requested Transform between %s and %s" % (req.frame_id, req.child_frame_id))
        requested_transform = TransformStamped()
        try:
            requested_transform = self.tf_buffer_.lookup_transform(req.frame_id, req.child_frame_id, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().error("An error occurred while transforming %s and %s: %s" %
                         (req.frame_id, req.child_frame_id, e))
            res.success = False
            return res
        
        res.transform = requested_transform
        res.success = True
        return res
    

def main():
    rclpy.init()

    simple_tf_kinematics = SimpleTfKinematics()
    rclpy.spin(simple_tf_kinematics)
    
    simple_tf_kinematics.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()