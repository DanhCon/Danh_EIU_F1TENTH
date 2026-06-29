#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "std_msgs/msg/empty.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

// Define a point struct for convenience
struct Point2D {
    double x;
    double y;
};

struct Control {
    double v;
    double steer;
};

class MPPIController : public rclcpp::Node
{
public:
    MPPIController() : Node("mppi_real_controller_node")
    {
        // ROS Parameters / Hardcoded for Real Car
        L = 0.33;
        dt = 0.05;
        horizon = 30;
        num_samples = 500;
        noise_sigma_v = 1.0;
        noise_sigma_steer = 0.30;
        lambda_ = 50.0;

        max_speed = 2.5;
        min_speed = 0.0;
        max_steer = 0.418;

        w_track = 60.0;
        w_progress = 1.5;
        w_control = 1.5;
        w_obstacle = 500.0;
        w_speed = 5.0;
        w_heading = 15.0;

        robot_radius = 0.35;
        danger_radius = 1.2;
        target_speed = 2.0;

        min_speed_curve = 1.8;
        curve_threshold = 0.35;
        lookahead_wps = 15;
        wp_window = 50;

        // Init Arrays
        nominal_control.resize(horizon, {target_speed, 0.0});
        
        is_reversing = false;
        forward_min_obs_dist = 999.0;
        last_best_obs_cost = 0.0;
        stuck_start_time = 0.0;
        is_stuck = false;

        x0 = 0.0; y0 = 0.0; theta0 = 0.0; v_cur = 0.0;
        odom_received = false;

        car_frame = "base_link";
        map_frame = "map";

        // Setup TF
        tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

        // Sub/Pub
        sub_odom = this->create_subscription<nav_msgs::msg::Odometry>(
            "/pf/pose/odom", 10, std::bind(&MPPIController::odom_callback, this, std::placeholders::_1));
        sub_laser = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&MPPIController::lidar_callback, this, std::placeholders::_1));

        pub_drive = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        pub_best_traj = this->create_publisher<visualization_msgs::msg::Marker>("/mppi_best_trajectory", 10);
        pub_waypoints = this->create_publisher<visualization_msgs::msg::MarkerArray>("/publish_full_waypoint", 10);

        // Timer (20Hz)
        control_timer = this->create_wall_timer(
            std::chrono::milliseconds(50), std::bind(&MPPIController::control_loop, this));

        // Random Generator
        std::random_device rd;
        rng = std::mt19937(rd());

        // Load Waypoints
        std::string csv_path = "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv";
        load_waypoints(csv_path);
        publish_waypoints_marker();
        
        RCLCPP_INFO(this->get_logger(), "MPPI C++ Controller started.");
    }

private:
    double L, dt;
    int horizon, num_samples;
    double noise_sigma_v, noise_sigma_steer, lambda_;
    double max_speed, min_speed, max_steer;
    double w_track, w_progress, w_control, w_obstacle, w_speed, w_heading;
    double robot_radius, danger_radius, target_speed;
    double min_speed_curve, curve_threshold;
    int lookahead_wps, wp_window;

    std::vector<Control> nominal_control;
    std::vector<Point2D> waypoints;
    std::vector<double> waypoint_headings;
    std::vector<double> waypoint_curvatures;

    std::vector<Point2D> map_obstacles;
    rclcpp::Time obstacle_stamp;

    bool is_reversing;
    double forward_min_obs_dist, last_best_obs_cost;
    double stuck_start_time;
    bool is_stuck;
    double reverse_end_time;

    double x0, y0, theta0, v_cur;
    bool odom_received;

    std::string car_frame, map_frame;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_laser;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr pub_drive;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_best_traj;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_waypoints;
    rclcpp::TimerBase::SharedPtr control_timer;

    std::mt19937 rng;

    // Helper functions
    void publish_waypoints_marker() {
        if (waypoints.empty()) return;
        
        visualization_msgs::msg::MarkerArray marker_array;
        
        // Points Marker
        visualization_msgs::msg::Marker points_marker;
        points_marker.header.frame_id = map_frame;
        points_marker.header.stamp = this->now();
        points_marker.ns = "waypoints";
        points_marker.id = 0;
        points_marker.type = visualization_msgs::msg::Marker::POINTS;
        points_marker.action = visualization_msgs::msg::Marker::ADD;
        points_marker.scale.x = 0.05;
        points_marker.scale.y = 0.05;
        points_marker.color.a = 1.0;
        points_marker.color.r = 1.0;
        points_marker.color.g = 1.0;
        points_marker.color.b = 0.0; // Yellow points
        
        for (const auto& wp : waypoints) {
            geometry_msgs::msg::Point p;
            p.x = wp.x;
            p.y = wp.y;
            p.z = 0.0;
            points_marker.points.push_back(p);
        }
        marker_array.markers.push_back(points_marker);
        pub_waypoints->publish(marker_array);
        RCLCPP_INFO(this->get_logger(), "Published full waypoint marker.");
    }

    double normalize_angle(double angle) {
        angle = std::fmod(angle + M_PI, 2.0 * M_PI);
        if (angle < 0) angle += 2.0 * M_PI;
        return angle - M_PI;
    }

    void load_waypoints(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open CSV: %s", path.c_str());
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            std::string val1, val2;
            if (std::getline(ss, val1, ',') && std::getline(ss, val2, ',')) {
                try {
                    Point2D pt;
                    pt.x = std::stod(val1);
                    pt.y = std::stod(val2);
                    waypoints.push_back(pt);
                } catch (const std::invalid_argument& e) {
                    continue; // Skip lines with invalid numbers (like headers)
                } catch (const std::out_of_range& e) {
                    continue;
                }
            }
        }
        
        int w = waypoints.size();
        waypoint_headings.resize(w);
        waypoint_curvatures.resize(w);
        
        if (w > 1) {
            std::vector<double> ds(w);
            for (int i = 0; i < w; i++) {
                int next = (i + 1) % w;
                double dx = waypoints[next].x - waypoints[i].x;
                double dy = waypoints[next].y - waypoints[i].y;
                waypoint_headings[i] = std::atan2(dy, dx);
                ds[i] = std::hypot(dx, dy);
                if (ds[i] < 1e-3) ds[i] = 1.0;
            }
            for (int i = 0; i < w; i++) {
                int next = (i + 1) % w;
                double hdg_diff = normalize_angle(waypoint_headings[next] - waypoint_headings[i]);
                waypoint_curvatures[i] = std::abs(hdg_diff) / ds[i];
            }
        }
        RCLCPP_INFO(this->get_logger(), "Loaded %d waypoints.", w);
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        x0 = msg->pose.pose.position.x;
        y0 = msg->pose.pose.position.y;
        
        auto q = msg->pose.pose.orientation;
        double siny = 2.0 * (q.w * q.z + q.x * q.y);
        double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        theta0 = std::atan2(siny, cosy);
        
        v_cur = msg->twist.twist.linear.x;
        odom_received = true;
    }

    void lidar_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        std::vector<Point2D> valid_pts;
        double min_fwd = 999.0;
        
        double angle = msg->angle_min;
        for (size_t i = 0; i < msg->ranges.size(); i++, angle += msg->angle_increment) {
            double r = msg->ranges[i];
            if (std::isfinite(r) && r > 0.15 && r < 4.5) {
                if (std::abs(angle) < 40.0 * M_PI / 180.0) {
                    if (r < min_fwd) min_fwd = r;
                }
                valid_pts.push_back({r * std::cos(angle), r * std::sin(angle)});
            }
        }
        forward_min_obs_dist = min_fwd;

        // Subsample (simple max 80 for speed)
        std::vector<Point2D> subsampled;
        int step = std::max(1, (int)(valid_pts.size() / 80));
        for (size_t i = 0; i < valid_pts.size(); i += step) {
            subsampled.push_back(valid_pts[i]);
        }

        // Transform to map
        geometry_msgs::msg::TransformStamped t;
        try {
            t = tf_buffer->lookupTransform(map_frame, msg->header.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.02));
        } catch (const tf2::TransformException & ex) {
            return;
        }

        double tx = t.transform.translation.x;
        double ty = t.transform.translation.y;
        auto q = t.transform.rotation;
        double yaw = std::atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z));
        double cos_y = std::cos(yaw);
        double sin_y = std::sin(yaw);

        map_obstacles.clear();
        for (const auto& p : subsampled) {
            double mx = cos_y * p.x - sin_y * p.y + tx;
            double my = sin_y * p.x + cos_y * p.y + ty;
            map_obstacles.push_back({mx, my});
        }
        obstacle_stamp = this->now();
    }

    void control_loop() {
        if (!odom_received || waypoints.empty()) {
            publish_drive(0.0, 0.0);
            return;
        }

        double now_s = this->now().seconds();
        
        // TTL for Lidar
        bool lidar_lost = false;
        if (obstacle_stamp.nanoseconds() == 0 || (now_s - obstacle_stamp.seconds() > 0.5)) {
            map_obstacles.clear();
            forward_min_obs_dist = 999.0;
            lidar_lost = true;
        }

        double min_obs_dist = 999.0;
        for (const auto& o : map_obstacles) {
            double d = std::hypot(o.x - x0, o.y - y0);
            if (d < min_obs_dist) min_obs_dist = d;
        }

        // Hysteresis & Anti-Stuck
        bool front_blocked = forward_min_obs_dist < 0.2;
        bool side_blocked = (min_obs_dist < 0.45 && std::abs(v_cur) < 0.1);
        
        if (side_blocked) {
            if (stuck_start_time == 0.0) stuck_start_time = now_s;
            else if (now_s - stuck_start_time > 0.8) is_stuck = true;
        } else {
            stuck_start_time = 0.0;
            is_stuck = false;
        }

        if (!is_reversing) {
            if (front_blocked || is_stuck) {
                is_reversing = true;
                reverse_end_time = now_s + 3.2;
                RCLCPP_WARN(this->get_logger(), "[SAFETY] Reverse Activated!");
            }
        } else {
            if (now_s > reverse_end_time && forward_min_obs_dist > 1.2 && !is_stuck) {
                is_reversing = false;
                RCLCPP_INFO(this->get_logger(), "[SAFETY] Back to Forward.");
            }
        }

        double dynamic_min_speed = is_reversing ? -0.8 : 0.0;
        double current_target_speed = target_speed;

        // Curvature Speed Profiling
        int nearest_wp = 0;
        double min_dist_wp = 9999.0;
        for (size_t i = 0; i < waypoints.size(); i++) {
            double d = std::hypot(waypoints[i].x - x0, waypoints[i].y - y0);
            if (d < min_dist_wp) { min_dist_wp = d; nearest_wp = i; }
        }

        double max_curve = 0.0;
        for (int i = 0; i < lookahead_wps; i++) {
            int idx = (nearest_wp + i) % waypoints.size();
            if (waypoint_curvatures[idx] > max_curve) max_curve = waypoint_curvatures[idx];
        }
        
        double speed_factor = std::max(0.0, std::min(1.0, 1.0 - std::pow(max_curve / curve_threshold, 2)));
        current_target_speed = min_speed_curve + (target_speed - min_speed_curve) * speed_factor;

        // Obstacle distance profiling
        double obs_speed_factor = 1.0;
        if (last_best_obs_cost >= 1.0) {
            double safe_braking_dist = std::max(1.5, v_cur * 0.5 + 0.5);
            double min_safe_dist = 0.6;
            if (forward_min_obs_dist < safe_braking_dist) {
                double span = std::max(0.2, safe_braking_dist - min_safe_dist);
                obs_speed_factor = std::max(0.0, std::min(1.0, (forward_min_obs_dist - min_safe_dist) / span));
            }
        }
        current_target_speed = std::max(0.0, current_target_speed * obs_speed_factor);

        if (lidar_lost) current_target_speed = 0.5;

        if (is_reversing) {
            current_target_speed = -0.8;
            if (nominal_control[0].v > -0.2) {
                for (auto& c : nominal_control) c.v = -0.8;
            }
        }

        double dynamic_max_speed = max_speed;
        if (last_best_obs_cost > 0.0) {
            dynamic_max_speed = 2.0;
            current_target_speed = std::min(current_target_speed, 2.0);
        }

        // MPPI Preparation
        std::normal_distribution<double> dist_v(0.0, noise_sigma_v);
        std::normal_distribution<double> dist_s(0.0, noise_sigma_steer);

        std::vector<double> costs(num_samples, 0.0);
        std::vector<std::vector<Control>> noise(num_samples, std::vector<Control>(horizon));
        
        for (int n = 0; n < num_samples; n++) {
            for (int t = 0; t < horizon; t++) {
                noise[n][t].v = dist_v(rng);
                noise[n][t].steer = dist_s(rng);
            }
        }

        // Extract local waypoints to small array for fast CPU cache access
        std::vector<Point2D> local_wps;
        std::vector<double> local_hdgs;
        std::vector<int> local_idxs;
        for (int i = -15; i < wp_window - 15; i++) {
            int idx = (nearest_wp + i) % waypoints.size();
            if (idx < 0) idx += waypoints.size();
            local_wps.push_back(waypoints[idx]);
            local_hdgs.push_back(waypoint_headings[idx]);
            local_idxs.push_back(idx);
        }

        double w_hdg_eff = is_reversing ? 0.0 : w_heading;
        double w_prog_eff = is_reversing ? 0.0 : w_progress;

        // MPPI PARALLEL ROLLOUT
        #pragma omp parallel for
        for (int n = 0; n < num_samples; n++) {
            double x = x0, y = y0, th = theta0;
            double track_cost = 0, speed_cost = 0, smooth_cost = 0;
            double hdg_cost = 0, prog_cost = 0, term_cost = 0, obs_cost = 0;
            
            bool collision_any = false;
            double col_count = 0.0, soft_obs = 0.0;
            
            double last_v = nominal_control[0].v;
            double last_steer = nominal_control[0].steer;

            for (int t = 0; t < horizon; t++) {
                double pert_v = nominal_control[t].v + noise[n][t].v;
                double pert_s = nominal_control[t].steer + noise[n][t].steer;
                
                pert_v = std::max(dynamic_min_speed, std::min(dynamic_max_speed, pert_v));
                pert_s = std::max(-max_steer, std::min(max_steer, pert_s));
                
                // Effective noise
                noise[n][t].v = pert_v - nominal_control[t].v;
                noise[n][t].steer = pert_s - nominal_control[t].steer;
                
                // Step
                x += pert_v * std::cos(th) * dt;
                y += pert_v * std::sin(th) * dt;
                th += (pert_v * std::tan(pert_s) / L) * dt;
                
                // Track cost
                double min_wp_d = 9999.0;
                int min_wi = 0;
                for (size_t wi = 0; wi < local_wps.size(); wi++) {
                    double d2 = std::pow(x - local_wps[wi].x, 2) + std::pow(y - local_wps[wi].y, 2);
                    if (d2 < min_wp_d) { min_wp_d = d2; min_wi = wi; }
                }
                track_cost += min_wp_d / horizon;
                
                // Speed cost
                speed_cost += std::pow(pert_v - current_target_speed, 2) / horizon;
                
                // Smooth cost
                if (t > 0) {
                    smooth_cost += (std::pow(pert_v - last_v, 2) + std::pow(pert_s - last_steer, 2)) / horizon;
                }
                last_v = pert_v; last_steer = pert_s;
                
                // Heading cost
                double err = normalize_angle(th - local_hdgs[min_wi]);
                hdg_cost += std::pow(err, 2) / horizon;
                
                // Obs cost
                double min_obs_d = 999.0;
                for (const auto& o : map_obstacles) {
                    double d = std::hypot(x - o.x, y - o.y);
                    if (d < min_obs_d) min_obs_d = d;
                }
                if (min_obs_d < robot_radius) {
                    collision_any = true;
                    col_count += 1.0;
                }
                if (min_obs_d < danger_radius && min_obs_d >= robot_radius) {
                    double sigma = (danger_radius - robot_radius) / 2.0;
                    soft_obs += std::exp(-0.5 * std::pow((min_obs_d - robot_radius)/sigma, 2)) / horizon;
                }
                
                // Terminal & Progress cost
                if (t == horizon - 1) {
                    term_cost = 3.0 * w_track * min_wp_d;
                    int global_idx = local_idxs[min_wi];
                    int prog_raw = (global_idx - local_idxs[15]) % (int)waypoints.size();
                    if (prog_raw < 0) prog_raw += waypoints.size();
                    if (prog_raw > (int)waypoints.size()/2) prog_raw -= waypoints.size();
                    prog_cost = -(double)prog_raw;
                }
            }
            
            obs_cost = (collision_any ? 100.0 : 0.0) + col_count * 10.0 + 0.1 * soft_obs;
            
            costs[n] = w_track * track_cost + w_prog_eff * prog_cost + 
                       w_control * smooth_cost + w_speed * speed_cost + 
                       w_hdg_eff * hdg_cost + w_obstacle * obs_cost + term_cost;
        }

        // Update MPPI
        double min_cost = costs[0];
        for (int n = 1; n < num_samples; n++) {
            if (costs[n] < min_cost) { min_cost = costs[n]; }
        }

        double w_sum = 0.0;
        std::vector<double> weights(num_samples);
        for (int n = 0; n < num_samples; n++) {
            weights[n] = std::exp(-(costs[n] - min_cost) / lambda_);
            w_sum += weights[n];
        }

        for (int t = 0; t < horizon; t++) {
            double v_sum = 0.0, s_sum = 0.0;
            for (int n = 0; n < num_samples; n++) {
                v_sum += (weights[n] / w_sum) * noise[n][t].v;
                s_sum += (weights[n] / w_sum) * noise[n][t].steer;
            }
            nominal_control[t].v = std::max(dynamic_min_speed, std::min(dynamic_max_speed, nominal_control[t].v + v_sum));
            nominal_control[t].steer = std::max(-max_steer, std::min(max_steer, nominal_control[t].steer + s_sum));
        }

        double opt_speed = nominal_control[0].v;
        double opt_steer = nominal_control[0].steer;

        // Shift horizon
        double last_s = nominal_control[horizon-1].steer;
        for (int t = 0; t < horizon - 1; t++) {
            nominal_control[t] = nominal_control[t+1];
        }
        nominal_control[horizon-1].v = current_target_speed;
        nominal_control[horizon-1].steer = last_s;

        publish_drive(opt_speed, opt_steer);
    }

    void publish_drive(double v, double steer) {
        auto msg = ackermann_msgs::msg::AckermannDriveStamped();
        msg.header.stamp = this->now();
        msg.header.frame_id = car_frame;
        msg.drive.speed = v;
        msg.drive.steering_angle = steer;
        pub_drive->publish(msg);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MPPIController>();
    // Use multithreaded executor
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}
