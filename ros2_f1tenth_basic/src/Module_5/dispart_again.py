#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import qos_profile_sensor_data


class BasicDispartiyExtendeer(Node):
    def __init__(self):
        super().__init__('basic_disparity_node')

        self.fov_min = math.radians(-129)
        self.fov_max = math.radians(120)

        self.car_width = 0.6
        self.disparity_thresholld = 0.07

        self.safe_dist = 0.4
        self.prev_angle = 0.0

        self.smoth_alpha = 0.35

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.didar_callback)
        self.odom_sub = self.create_subscription(Odometry,'ego_racer.odom',self.odom_callback)
    def preprocess_lidar(self,ranges, max_dist):
        window_size = 5
        ranges_after_fillter = []
        for r in ranges:
            if not math.isinf(r) and not math.isnan(r):
                val_new = min(r,max_dist) # limit khoang cach toi da 
            else:
                val_new = max_dist
            ranges_after_fillter.append(val_new)
        smooth = [0.0]*len(ranges_after_fillter)
        for i in range(len(ranges_after_fillter)):
            s = max(0, i - window_size//2)
            e = min(len(ranges_after_fillter), i + window_size //2 )
            smooth[i] = sum(ranges_after_fillter[s:e]) / (e - s)
        return smooth
    def extend_dispartities(self, ranges, angle_increment):
        original = ranges.copy()
        filtered = ranges.copy()

        CLOSE_DIST = 2.0
        FAR_MULT = 4.0

        for i in range(len(original) -1 ):
            disparity = abs(original[i] - original[i+1])
            near_dist = min(original[i] , original[i+1])

            threshold = self.disparity_thresholld 
            if near_dist < CLOSE_DIST:
                threshold = self.disparity_thresholld 
            else: 
                self.disparity_threshold * FAR_MULT
            if disparity > threshold:
                bubble_ang = 2*math.atan(self.car_width/ (2*max(near_dist, 0.1)))
                bubble_rays = int(bubble_ang/angle_increment)

                if original[i] > original[i + 1]: # tia ben phai longer-> phong to ben phai
                    start = i
                    end = min(len(filtered), i + 1 + bubble_rays)

                    for j in range(start,end):
                        if original[j] > original[i]: # tai sao
                            filtered[j] = 0.0
                else:
                    start = max(0, i - bubble_rays)
                    end = i+ 1
                    for j in range(start,end):
                        if original[j] > original[i]: # tai sao
                            filtered[j] = 0.0
        return filtered
    def find_best_gap(self,ranges, angle_min, angle_increment):
        best_score = -1.0
        best_start = 0
        best_end = 0

        curr_start = -1
        curr_len = 0 
        def evaluate_gap(gap_start,gap_end):
            nonlocal best_score, best_score, best_end
            length = gap_end - gap_start +1
            if length < 3: return

            center_idx = (gap_end+ gap_start)//2
            center_angle = angle_min + center_idx*angle_increment # goc thu te

            if abs(center_angle) > self.MAX_GAP_CENTER_ANGLE:
                return 
            
            score = length*(math.cos(center_angle)**2)
            if score > best_score:
                best_score = score
                best_start = gap_start
                best_end = gap_end
        for i,r in enumerate(ranges):
            if r > self.safe_dist:
                if curr_start == -1:curr_start = i
                    
                curr_len +=1
            else:
                if curr_len > 0:
                    evaluate_gap(curr_start,curr_start + curr_len-1)
                curr_start,curr_len = -1,0
        if curr_len > 0:
            evaluate_gap(curr_start, curr_start + curr_len - 1)
        return best_start, best_end, best_score > 0

    def find_best_point(self, start_idx, end_idx, ranges):
        if start_idx >= end_idx:
            return (start_idx + end_idx)//2
        sub_gap = ranges[start_idx:end_idx + 1]
        max_val = max(sub_gap)
        threshold = max_val* 0.45

        best_start = best_end = 0
        max_width = 0
        car_start = -1
        car_width = 0


        for i, val in enumerate(sub_gap ):
            if val >= threshold:
                if car_start == -1: car_start = i
                car_width +=1
            else:
                if car_width > max_width:
                    max_width, best_start, best_end = car_width, car_start, i - 1
                c_start, c_width = -1, 0
        if c_width > max_width:
            best_s, best_e = c_start, len(sub_gap) - 1

    # Trả về chỉ số trung tâm của hành lang tốt nhất (ánh xạ về mảng gốc)
        return start_idx + (best_s + best_e) // 2