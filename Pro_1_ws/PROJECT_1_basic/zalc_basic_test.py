import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
import tf2_ros
import math
import time

from zlac8015d_driver import ZLAC8015D_Driver


class ZLAC8015DTest(Node):
    def __init__(self):
        super().__init__('ZLAC_Test')