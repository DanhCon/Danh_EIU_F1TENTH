#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <omp.h>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

struct Point2D { double x, y; };
struct Control { double v, steer; };

class MPPIController : public rclcpp::Node {
public:
    MPPIController() : Node("mppi_sim_controller_node") {
        this->declare_parameter("horizon", 30);
        this->declare_parameter("num_samples", 500);
        this->declare_parameter("dt", 0.05);
        
        horizon = this->get_parameter("horizon").as_int();
        num_samples = this->get_parameter("num_samples").as_int();
        dt = this->get_parameter("dt").as_double();
        
        lambda_ = 50.0;
        
        w_track = 20.0;
        w_progress = 5.0;
        w_heading = 10.0;
        w_obs = 500.0;
        w_smooth = 1.5;
        w_speed = 3.0;

        // Pre-allocate buffers (Bug 4 - Performance)
        noise_buf.resize(num_samples, std::vector<Control>(horizon));
        costs_buf.resize(num_samples, 0.0);
        weights_buf.resize(num_samples, 0.0);
        nominal_control.resize(horizon, {0.0, 0.0});

        car_frame = "ego_racecar/base_link";
        map_frame = "map";

        tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

        sub_odom = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10, std::bind(&MPPIController::odom_callback, this, std::placeholders::_1));
        sub_laser = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&MPPIController::lidar_callback, this, std::placeholders::_1));

        pub_drive = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        pub_best_traj = this->create_publisher<visualization_msgs::msg::Marker>("/mppi_best_trajectory", 10);
        pub_waypoints = this->create_publisher<visualization_msgs::msg::MarkerArray>("/publish_full_waypoint", 10);

        control_timer = this->create_wall_timer(
            std::chrono::milliseconds((int)(dt * 1000)), std::bind(&MPPIController::control_loop, this));

        std::random_device rd;
        rng = std::mt19937(rd());

        std::string csv_path = "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv";
        load_waypoints(csv_path);
        publish_waypoints_marker();
        
        RCLCPP_INFO(this->get_logger(), "MPPI C++ Sim Controller started.");
    }

private:
    int horizon, num_samples;
    double dt, lambda_;
    double w_track, w_progress, w_heading, w_obs, w_smooth, w_speed;
    
    std::vector<Point2D> waypoints;
    std::vector<double> waypoint_headings;
    std::vector<double> waypoint_curvatures;
    
    std::vector<Control> nominal_control;
    std::vector<std::vector<Control>> noise_buf;
    std::vector<double> costs_buf;
    std::vector<double> weights_buf;
    
    double x0 = 0.0, y0 = 0.0, theta0 = 0.0, v_cur = 0.0;
    bool odom_received = false;
    rclcpp::Time odom_stamp; // Bug 7 - Logic
    
    std::vector<Point2D> map_obstacles;
    rclcpp::Time obstacle_stamp;
    std::mutex obs_mutex; // Bug 3 - Performance
    
    std::string car_frame, map_frame;
    
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_laser;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr pub_drive;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_best_traj;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_waypoints;
    rclcpp::TimerBase::SharedPtr control_timer;
    
    std::unique_ptr<tf2_ros::Buffer> tf_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener;
    std::mt19937 rng;

    bool is_reversing = false;
    bool is_stuck_timer_active = false; // Bug 3 - Logic
    double stuck_start_time = 0.0;
    double reverse_end_time = 0.0; // Bug 4 - Logic
    
    double last_best_obs_cost = 0.0;
    double last_target_speed = 0.0;
    int last_nearest_wp_idx = 15; // Bug 5 - Performance

    void publish_waypoints_marker() {
        if (waypoints.empty()) return;
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker points_marker;
        points_marker.header.frame_id = map_frame;
        points_marker.header.stamp = this->now();
        points_marker.ns = "waypoints";
        points_marker.id = 0;
        points_marker.type = visualization_msgs::msg::Marker::POINTS;
        points_marker.action = visualization_msgs::msg::Marker::ADD;
        points_marker.scale.x = 0.05; points_marker.scale.y = 0.05;
        points_marker.color.a = 1.0; points_marker.color.r = 1.0;
        points_marker.color.g = 1.0; points_marker.color.b = 0.0;
        
        for (const auto& wp : waypoints) {
            geometry_msgs::msg::Point p;
            p.x = wp.x; p.y = wp.y; p.z = 0.0;
            points_marker.points.push_back(p);
        }
        marker_array.markers.push_back(points_marker);
        pub_waypoints->publish(marker_array);
    }

    double normalize_angle(double angle) {
        angle = std::fmod(angle + M_PI, 2.0 * M_PI);
        if (angle < 0) angle += 2.0 * M_PI;
        return angle - M_PI;
    }

    void load_waypoints(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return;
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            std::string val1, val2;
            if (std::getline(ss, val1, ',') && std::getline(ss, val2, ',')) {
                try {
                    Point2D pt; pt.x = std::stod(val1); pt.y = std::stod(val2);
                    waypoints.push_back(pt);
                } catch (...) { continue; }
            }
        }
        int w = waypoints.size();
        waypoint_headings.resize(w);
        waypoint_curvatures.resize(w);
        for (int i = 0; i < w; i++) {
            Point2D p1 = waypoints[(i - 1 + w) % w];
            Point2D p2 = waypoints[i];
            Point2D p3 = waypoints[(i + 1) % w];
            waypoint_headings[i] = std::atan2(p3.y - p1.y, p3.x - p1.x);
            double dx1 = p2.x - p1.x; double dy1 = p2.y - p1.y;
            double dx2 = p3.x - p2.x; double dy2 = p3.y - p2.y;
            double area = dx1 * dy2 - dy1 * dx2;
            double len1 = std::hypot(dx1, dy1);
            double len2 = std::hypot(dx2, dy2);
            double len3 = std::hypot(p3.x - p1.x, p3.y - p1.y);
            if (len1 * len2 * len3 == 0) waypoint_curvatures[i] = 0.0;
            else waypoint_curvatures[i] = 4.0 * area / (len1 * len2 * len3);
        }
    }

    void lidar_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        geometry_msgs::msg::TransformStamped t;
        try {
            t = tf_buffer->lookupTransform(map_frame, msg->header.frame_id, tf2::TimePointZero);
        } catch (...) { return; }
        
        auto& q = t.transform.rotation;
        double tf_yaw = std::atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z));
        
        std::vector<Point2D> temp_obs;
        double angle_step_deg = 1.0;
        int step = std::max(1, (int)(angle_step_deg * M_PI / 180.0 / msg->angle_increment));
        
        for (size_t i = 0; i < msg->ranges.size(); i += step) {
            double angle = msg->angle_min + i * msg->angle_increment;
            double r = msg->ranges[i];
            if (std::isnormal(r) && r > 0.1 && r < 3.5) {
                double px = r * std::cos(angle);
                double py = r * std::sin(angle);
                double mx = t.transform.translation.x + px * std::cos(tf_yaw) - py * std::sin(tf_yaw);
                double my = t.transform.translation.y + px * std::sin(tf_yaw) + py * std::cos(tf_yaw);
                temp_obs.push_back({mx, my});
            }
        }
        
        std::lock_guard<std::mutex> lock(obs_mutex);
        map_obstacles = temp_obs;
        obstacle_stamp = this->now();
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        v_cur = msg->twist.twist.linear.x;
        odom_received = true;
        odom_stamp = this->now();
    }

    void control_loop() {
        double now_s = this->now().seconds();
        if (!odom_received || waypoints.empty() || (now_s - odom_stamp.seconds() > 0.5)) {
            publish_drive(0.0, 0.0);
            return;
        }

        // SIM Specific: Get x, y, theta from TF
        geometry_msgs::msg::TransformStamped tf_map_to_base;
        try {
            tf_map_to_base = tf_buffer->lookupTransform(map_frame, car_frame, tf2::TimePointZero);
        } catch (const tf2::TransformException & ex) {
            return;
        }
        x0 = tf_map_to_base.transform.translation.x;
        y0 = tf_map_to_base.transform.translation.y;
        auto q = tf_map_to_base.transform.rotation;
        double siny = 2.0 * (q.w * q.z + q.x * q.y);
        double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        theta0 = std::atan2(siny, cosy);


        if (std::isnan(x0) || std::isnan(y0) || std::isnan(theta0)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "NaN in pose detected!");
            return;
        }
        // Snapshot obstacles (Bug 3 - Perf)
        std::vector<Point2D> obs_snapshot;
        rclcpp::Time obs_stamp;
        {
            std::lock_guard<std::mutex> lock(obs_mutex);
            obs_snapshot = map_obstacles;
            obs_stamp = obstacle_stamp;
        }

        // Local WP Search (Bug 5 - Perf)
        int nearest_wp = last_nearest_wp_idx;
        double min_d = 9999.0;
        int search_start = std::max(0, nearest_wp - 20);
        int search_end = std::min((int)waypoints.size(), nearest_wp + 40);
        for (int i = search_start; i < search_end; i++) {
            double d = std::hypot(waypoints[i].x - x0, waypoints[i].y - y0);
            if (d < min_d) { min_d = d; nearest_wp = i; }
        }
        last_nearest_wp_idx = nearest_wp;

        int wp_window = 50;
        std::vector<Point2D> local_wps;
        std::vector<double> local_hdgs;
        std::vector<int> local_idxs;
        for (int i = -15; i < wp_window - 15; i++) {
            int idx = (nearest_wp + i) % (int)waypoints.size();
            if (idx < 0) idx += waypoints.size();
            local_wps.push_back(waypoints[idx]);
            local_hdgs.push_back(waypoint_headings[idx]);
            local_idxs.push_back(idx);
        }
        int local_nearest_idx = -1;
        for (int i = 0; i < (int)local_idxs.size(); i++) {
            if (local_idxs[i] == nearest_wp) { local_nearest_idx = i; break; }
        }

        // Curvature Profiling
        double max_c = 0.0;
        for (int i = 0; i < 15; i++) {
            int idx = (nearest_wp + i) % waypoints.size();
            double c = std::abs(waypoint_curvatures[idx]);
            if (c > max_c) max_c = c;
        }
        
        double target_speed = 2.0;
        double min_speed_curve = 1.8;
        double curve_thresh = 0.35;
        double speed_factor = max_c > curve_thresh ? std::max(0.0, 1.0 - (max_c - curve_thresh)) : 1.0;
        double current_target_speed = min_speed_curve + (target_speed - min_speed_curve) * speed_factor;

        // Rate limit deceleration (Bug 6 - Logic)
        if (last_best_obs_cost > 0.0) {
            double obs_speed_factor = std::max(0.3, 1.0 - last_best_obs_cost / 200.0);
            current_target_speed = std::max(-0.3, current_target_speed * obs_speed_factor); 
        }
        double max_decel = 4.0;
        current_target_speed = std::max(last_target_speed - max_decel * dt, current_target_speed);
        last_target_speed = current_target_speed;

        double danger_radius = 1.2;
        bool front_blocked = false;
        // TTL Check (Bug 5 - Logic)
        if (obs_stamp.nanoseconds() != 0 && (now_s - obs_stamp.seconds() < 0.5)) {
            double braking_dist = std::max(0.8, v_cur * v_cur / 6.0 + 0.3); // v^2/(2a) + margin
            for (const auto& o : obs_snapshot) {
                double dx = o.x - x0;
                if (std::abs(dx) > std::max(danger_radius, braking_dist)) continue; 
                double dy = o.y - y0;
                if (std::abs(dy) > std::max(danger_radius, 0.35)) continue; 
                
                double dx_local = dx * std::cos(-theta0) - dy * std::sin(-theta0);
                double dy_local = dx * std::sin(-theta0) + dy * std::cos(-theta0);
                if (dx_local > 0.1 && dx_local < braking_dist && std::abs(dy_local) < 0.35) {
                    front_blocked = true;
                    break;
                }
            }
        }

        // Anti-stuck watchdog (Bug 3 - Logic)
        bool is_stuck = false;
        double last_s = nominal_control[0].steer;
        if (v_cur < 0.1 && std::abs(last_s) > 0.3) {
            if (!is_stuck_timer_active) {
                stuck_start_time = now_s;
                is_stuck_timer_active = true;
            } else if (now_s - stuck_start_time > 0.8) {
                is_stuck = true;
            }
        } else {
            is_stuck_timer_active = false;
        }

        // Reverse logic (Bug 1 - Logic)
        if (!is_reversing && (front_blocked || is_stuck)) {
            is_reversing = true;
            reverse_end_time = now_s + 3.2;
            for (auto& c : nominal_control) { c.v = -0.8; c.steer = 0.0; } // Flush ONCE
        }
        if (is_reversing && now_s > reverse_end_time) {
            is_reversing = false;
            is_stuck_timer_active = false;
            for (auto& c : nominal_control) { c.v = current_target_speed; c.steer = 0.0; } // Flush ONCE
        }

        double dynamic_min_speed = is_reversing ? -1.5 : -0.3; // Bug 2 - Logic
        double dynamic_max_speed = is_reversing ? -0.3 : std::max(0.5, current_target_speed);
        if (is_reversing) current_target_speed = -0.8;

        double w_hdg_eff = w_heading; 
        double w_prog_eff = is_reversing ? 0.0 : w_progress;

        std::normal_distribution<double> dist_v(0.0, 1.0);
        std::normal_distribution<double> dist_s(0.0, 0.4);
        for (int n = 0; n < num_samples; n++) {
            for (int t = 0; t < horizon; t++) {
                noise_buf[n][t].v = dist_v(rng);
                noise_buf[n][t].steer = dist_s(rng);
            }
        }

        #pragma omp parallel for
        for (int n = 0; n < num_samples; n++) {
            double x = x0, y = y0, th = theta0;
            double cost = 0.0;
            double track_cost = 0.0, prog_cost = 0.0, obs_cost = 0.0;
            double smooth_cost = 0.0, speed_cost = 0.0, hdg_cost = 0.0;
            
            int global_idx = nearest_wp;
            double prev_pert_v = nominal_control[0].v;
            double prev_pert_s = nominal_control[0].steer;

            for (int t = 0; t < horizon; t++) {
                double orig_noise_v = noise_buf[n][t].v; // Bug 1 - Math
                double orig_noise_s = noise_buf[n][t].steer;
                
                double pert_v = std::max(dynamic_min_speed, std::min(dynamic_max_speed, nominal_control[t].v + orig_noise_v));
                double pert_s = std::max(-0.418, std::min(0.418, nominal_control[t].steer + orig_noise_s));

                x += pert_v * std::cos(th) * dt;
                y += pert_v * std::sin(th) * dt;
                th += pert_v * std::tan(pert_s) / 0.33 * dt;
                th = normalize_angle(th);

                double min_wp_d2 = 999.0;
                int min_wi = 0;
                for (size_t wi = 0; wi < local_wps.size(); wi++) {
                    double d2 = std::pow(x - local_wps[wi].x, 2) + std::pow(y - local_wps[wi].y, 2);
                    if (d2 < min_wp_d2) { min_wp_d2 = d2; min_wi = wi; }
                }
                track_cost += min_wp_d2; // Bug 2 - Perf
                global_idx = local_idxs[min_wi];

                double ref_hdg = is_reversing ? normalize_angle(local_hdgs[min_wi] + M_PI) : local_hdgs[min_wi]; // Bug 4 - Logic
                double err = normalize_angle(th - ref_hdg);
                hdg_cost += std::pow(err, 2);

                speed_cost += std::pow(pert_v - current_target_speed, 2);

                if (t > 0) {
                    smooth_cost += std::pow(pert_s - prev_pert_s, 2) + std::pow(pert_v - prev_pert_v, 2);
                }
                prev_pert_v = pert_v;
                prev_pert_s = pert_s;

                double min_d = 999.0;
                for (const auto& o : obs_snapshot) {
                    double dx = x - o.x;
                    if (std::abs(dx) > danger_radius) continue; // Bug 1 - Perf
                    double dy = y - o.y;
                    if (std::abs(dy) > danger_radius) continue; // Bug 1 - Perf
                    double d = std::hypot(dx, dy);
                    if (d < min_d) min_d = d;
                }
                if (min_d < danger_radius) obs_cost += std::pow(danger_radius - min_d, 2);
                if (min_d < 0.2) obs_cost += 1000.0;

                if (t == horizon - 1) {
                    double term_cost = 3.0 * w_track * min_wp_d2;
                    term_cost += 3.0 * w_hdg_eff * std::pow(err, 2); // Bug 8 - Logic
                    
                    int prog_raw = (global_idx - local_idxs[local_nearest_idx]) % (int)waypoints.size();
                    if (prog_raw < 0) prog_raw += waypoints.size(); // Bug 3 - Math
                    double prog_cost_val = -(double)prog_raw;
                    if (prog_raw > (int)waypoints.size() / 2) {
                        prog_cost_val = (double)(waypoints.size() - prog_raw);
                    }
                    prog_cost += prog_cost_val;
                    cost += term_cost;
                }
            }
            
            cost += w_track * track_cost + w_hdg_eff * hdg_cost + w_speed * speed_cost +
                    w_smooth * smooth_cost + w_obs * obs_cost + w_prog_eff * prog_cost;
            costs_buf[n] = cost;
        }

        double min_cost = costs_buf[0];
        int best_idx = 0;
        for (int n = 1; n < num_samples; n++) {
            if (costs_buf[n] < min_cost) { min_cost = costs_buf[n]; best_idx = n; }
        }

        // Bug 5 (Logic) Update last_best_obs_cost based on best traj
        double best_obs_cost = 0.0;
        double x_b = x0, y_b = y0, th_b = theta0;
        for (int t = 0; t < horizon; t++) {
            double orig_v = noise_buf[best_idx][t].v;
            double orig_s = noise_buf[best_idx][t].steer;
            double pert_v = std::max(dynamic_min_speed, std::min(dynamic_max_speed, nominal_control[t].v + orig_v));
            double pert_s = std::max(-0.418, std::min(0.418, nominal_control[t].steer + orig_s));
            double min_d = 999.0;
            for (const auto& o : obs_snapshot) {
                double dx = x_b - o.x;
                if (std::abs(dx) > danger_radius) continue;
                double dy = y_b - o.y;
                if (std::abs(dy) > danger_radius) continue;
                double d = std::hypot(dx, dy);
                if (d < min_d) min_d = d;
            }
            if (min_d < danger_radius) best_obs_cost += std::pow(danger_radius - min_d, 2);
            x_b += pert_v * std::cos(th_b) * dt; 
            y_b += pert_v * std::sin(th_b) * dt;
            th_b += pert_v * std::tan(pert_s) / 0.33 * dt;
            th_b = normalize_angle(th_b);
        }
        last_best_obs_cost = best_obs_cost;

        double w_sum = 0.0;
        for (int n = 0; n < num_samples; n++) {
            double w = std::exp(- (costs_buf[n] - min_cost) / lambda_);
            weights_buf[n] = w;
            w_sum += w;
        }

        for (int t = 0; t < horizon; t++) {
            double num_v = 0.0, num_s = 0.0;
            for (int n = 0; n < num_samples; n++) {
                num_v += weights_buf[n] * noise_buf[n][t].v; // Bug 1 (Math) correctly uses original noise
                num_s += weights_buf[n] * noise_buf[n][t].steer;
            }
            nominal_control[t].v += num_v / w_sum;
            nominal_control[t].steer += num_s / w_sum;
            
            nominal_control[t].v = std::max(dynamic_min_speed, std::min(dynamic_max_speed, nominal_control[t].v));
            nominal_control[t].steer = std::max(-0.418, std::min(0.418, nominal_control[t].steer));
        }

        publish_drive(nominal_control[0].v, nominal_control[0].steer);

        for (int t = 0; t < horizon - 1; t++) {
            nominal_control[t] = nominal_control[t+1];
        }
        nominal_control[horizon-1].v = current_target_speed; // Bug 6 - Logic Fix
        nominal_control[horizon-1].steer = nominal_control[horizon-2].steer * 0.5;
    }

    void publish_drive(double v, double steer) {
        if (std::isnan(v) || std::isnan(steer)) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "NaN detected in control! v=%f, steer=%f", v, steer);
            return;
        }
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp = this->now();
        drive_msg.header.frame_id = car_frame;
        drive_msg.drive.speed = v;
        drive_msg.drive.steering_angle = steer;
        pub_drive->publish(drive_msg);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MPPIController>();
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}
