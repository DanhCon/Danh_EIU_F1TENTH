#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#include "eiquadprog.hpp"

using namespace std::chrono_literals;

double normalize_angle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

struct Waypoint {
    double x, y, yaw;
};

class MPCNode : public rclcpp::Node {
public:
    MPCNode() : Node("mpc_controller_node") {
        m = 3.47;
        Iz = 0.04712;
        Caf = 60.0;
        Car = 60.0;
        lf = 0.158;
        lr = 0.171;
        Ts = 0.05;

        hz = 15;
        outputs = 2;
        inputs = 2;

        Q = Eigen::MatrixXd::Zero(2, 2);
        Q(0, 0) = 10.0; Q(1, 1) = 200.0;

        S = Eigen::MatrixXd::Zero(2, 2);
        S(0, 0) = 10.0; S(1, 1) = 200.0;

        R = Eigen::MatrixXd::Zero(2, 2);
        R(0, 0) = 8000.0; R(1, 1) = 50.0;

        delta_max = 0.35;
        delta_min = -0.35;
        du_delta_max = 0.05;

        a_max = 3.0;
        a_min = -4.0;
        du_a_max = 1.0;

        v_max = 2.0;
        v_min = 0.8;

        curvature_lookahead = 13;
        v_straight = 2.0;
        v_curve = 1.2;

        U1 = 0.0;
        U2 = 0.0;
        v_x_current = 1.0;

        start_index = -1;

        car_frame = "base_link";
        map_frame = "map";

        tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

        sub_odom = this->create_subscription<nav_msgs::msg::Odometry>(
            "/pf/pose/odom", 10, std::bind(&MPCNode::odom_callback, this, std::placeholders::_1));
        pub_drive = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        pub_marker_path = this->create_publisher<visualization_msgs::msg::MarkerArray>("/publish_full_waypoint", 10);
        pub_mpc_ref = this->create_publisher<visualization_msgs::msg::Marker>("/mpc_lookahead_points", 10);
        pub_mpc_predict = this->create_publisher<visualization_msgs::msg::Marker>("/mpc_predict_path", 10);

        std::string csv_path = "/home/fablab_01/danh_pp_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv";
        load_waypoints(csv_path);
        publish_full_waypoint();

        last_mpc_time = this->get_clock()->now();

        RCLCPP_INFO(this->get_logger(), "MPC C++ tối ưu đã khởi động!");
    }

private:
    double m, Iz, Caf, Car, lf, lr, Ts;
    int hz, outputs, inputs;
    Eigen::MatrixXd Q, S, R;
    double delta_max, delta_min, du_delta_max;
    double a_max, a_min, du_a_max;
    double v_max, v_min, curvature_lookahead, v_straight, v_curve;
    double U1, U2, v_x_current;
    
    int start_index;
    std::vector<Waypoint> waypoints;
    std::string car_frame, map_frame;
    
    rclcpp::Time last_mpc_time;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr pub_drive;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker_path;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_mpc_ref;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_mpc_predict;

    void calculate_state_space(double v_x, Eigen::MatrixXd& Ad, Eigen::MatrixXd& Bd, Eigen::MatrixXd& Cd) {
        v_x = std::max(v_x, 1.0);
        double A1 = -(2 * Caf + 2 * Car) / (m * v_x);
        double A2 = -v_x - (2 * Caf * lf - 2 * Car * lr) / (m * v_x);
        double A3 = -(2 * lf * Caf - 2 * lr * Car) / (Iz * v_x);
        double A4 = -(2 * lf * lf * Caf + 2 * lr * lr * Car) / (Iz * v_x);

        Eigen::MatrixXd A_c = Eigen::MatrixXd::Zero(5, 5);
        A_c(0, 0) = A1; A_c(0, 2) = A2;
        A_c(1, 2) = 1.0;
        A_c(2, 0) = A3; A_c(2, 2) = A4;
        A_c(3, 0) = 1.0; A_c(3, 1) = v_x;

        Eigen::MatrixXd B_c = Eigen::MatrixXd::Zero(5, 2);
        B_c(0, 0) = 2 * Caf / m;
        B_c(2, 0) = 2 * lf * Caf / Iz;
        B_c(4, 1) = 1.0;

        Eigen::MatrixXd C_c = Eigen::MatrixXd::Zero(2, 5);
        C_c(0, 1) = 1.0;
        C_c(1, 3) = 1.0;

        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(5, 5);
        Eigen::MatrixXd inv_term = (I - (Ts / 2.0) * A_c).inverse();
        Ad = inv_term * (I + (Ts / 2.0) * A_c);
        Bd = inv_term * (B_c * Ts);
        Cd = C_c;
    }

    void mpc_simplification(const Eigen::MatrixXd& Ad, const Eigen::MatrixXd& Bd, const Eigen::MatrixXd& Cd,
                            Eigen::MatrixXd& Hdb, Eigen::MatrixXd& Fdbt, Eigen::MatrixXd& Cdb, Eigen::MatrixXd& Adc) {
        int n_x = Ad.rows(); // 5
        int n_u = Bd.cols(); // 2
        int n_y = Cd.rows(); // 2
        int n_aug = n_x + n_u; // 7

        Eigen::MatrixXd A_aug = Eigen::MatrixXd::Zero(n_aug, n_aug);
        A_aug.topLeftCorner(n_x, n_x) = Ad;
        A_aug.topRightCorner(n_x, n_u) = Bd;
        A_aug.bottomRightCorner(n_u, n_u) = Eigen::MatrixXd::Identity(n_u, n_u);

        Eigen::MatrixXd B_aug = Eigen::MatrixXd::Zero(n_aug, n_u);
        B_aug.topRows(n_x) = Bd;
        B_aug.bottomRows(n_u) = Eigen::MatrixXd::Identity(n_u, n_u);

        Eigen::MatrixXd C_aug = Eigen::MatrixXd::Zero(n_y, n_aug);
        C_aug.topLeftCorner(n_y, n_x) = Cd;

        Eigen::MatrixXd CQC = C_aug.transpose() * Q * C_aug;
        Eigen::MatrixXd CSC = C_aug.transpose() * S * C_aug;
        Eigen::MatrixXd QC = Q * C_aug;
        Eigen::MatrixXd SC = S * C_aug;

        int s_x = n_aug * hz;
        int s_y = n_y * hz;
        int s_u = n_u * hz;

        Eigen::MatrixXd Qdb = Eigen::MatrixXd::Zero(s_x, s_x);
        Eigen::MatrixXd Tdb = Eigen::MatrixXd::Zero(s_y, s_x);
        Eigen::MatrixXd Rdb = Eigen::MatrixXd::Zero(s_u, s_u);
        Cdb = Eigen::MatrixXd::Zero(s_x, s_u);
        Adc = Eigen::MatrixXd::Zero(s_x, n_aug);

        Eigen::MatrixXd A_aug_p = A_aug;

        for (int i = 0; i < hz; i++) {
            if (i == hz - 1) {
                Qdb.block(n_aug * i, n_aug * i, n_aug, n_aug) = CSC;
                Tdb.block(n_y * i, n_aug * i, n_y, n_aug) = SC;
            } else {
                Qdb.block(n_aug * i, n_aug * i, n_aug, n_aug) = CQC;
                Tdb.block(n_y * i, n_aug * i, n_y, n_aug) = QC;
            }
            Rdb.block(n_u * i, n_u * i, n_u, n_u) = R;

            for (int j = 0; j <= i; j++) {
                Eigen::MatrixXd pw = Eigen::MatrixXd::Identity(n_aug, n_aug);
                for (int k = 0; k < i - j; k++) pw = pw * A_aug;
                Cdb.block(n_aug * i, n_u * j, n_aug, n_u) = pw * B_aug;
            }

            Adc.block(n_aug * i, 0, n_aug, n_aug) = A_aug_p;
            A_aug_p = A_aug_p * A_aug;
        }

        Hdb = Cdb.transpose() * Qdb * Cdb + Rdb;
        Eigen::MatrixXd temp1 = Adc.transpose() * Qdb * Cdb;
        Eigen::MatrixXd temp2 = -Tdb * Cdb;
        Fdbt = Eigen::MatrixXd::Zero(temp1.rows() + temp2.rows(), temp1.cols());
        Fdbt << temp1, temp2;
    }

    void build_constraints(Eigen::MatrixXd& G, Eigen::VectorXd& ht) {
        int n = hz;
        int n2 = 2 * n;
        Eigen::MatrixXd I2 = Eigen::MatrixXd::Identity(2, 2);

        Eigen::MatrixXd I_block = Eigen::MatrixXd::Zero(n2, n2);
        for (int k = 0; k < n; k++) I_block.block(2 * k, 2 * k, 2, 2) = I2;

        Eigen::VectorXd rate_max(n2);
        for (int k = 0; k < n; k++) {
            rate_max(2 * k) = du_delta_max;
            rate_max(2 * k + 1) = du_a_max;
        }

        Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n2, n2);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                L.block(2 * i, 2 * j, 2, 2) = I2;
            }
        }

        Eigen::VectorXd U_current(n2), U_max_vec(n2), U_min_vec(n2);
        for (int k = 0; k < n; k++) {
            U_current(2 * k) = U1;
            U_current(2 * k + 1) = U2;
            U_max_vec(2 * k) = delta_max;
            U_max_vec(2 * k + 1) = a_max;
            U_min_vec(2 * k) = delta_min;
            U_min_vec(2 * k + 1) = a_min;
        }

        G = Eigen::MatrixXd::Zero(4 * n2, n2);
        G << I_block, -I_block, L, -L;

        ht = Eigen::VectorXd::Zero(4 * n2);
        Eigen::VectorXd ht_mag_pos = U_max_vec - U_current;
        Eigen::VectorXd ht_mag_neg = U_current - U_min_vec;
        ht << rate_max, rate_max, ht_mag_pos, ht_mag_neg;
    }

    double compute_target_speed(int nearest_idx) {
        int n = waypoints.size();
        int k = curvature_lookahead;

        Waypoint p_prev = waypoints[(nearest_idx - k + n) % n];
        Waypoint p_curr = waypoints[nearest_idx];
        Waypoint p_next = waypoints[(nearest_idx + k) % n];

        double dx1 = p_curr.x - p_prev.x;
        double dy1 = p_curr.y - p_prev.y;
        double dx2 = p_next.x - p_curr.x;
        double dy2 = p_next.y - p_curr.y;

        double cross = std::abs(dx1 * dy2 - dy1 * dx2);
        double norm1 = std::hypot(dx1, dy1) + 1e-6;
        double norm2 = std::hypot(dx2, dy2) + 1e-6;
        double curvature = cross / (norm1 * norm2);

        double curvature_clipped = std::min(curvature, 1.0);
        double target_speed = v_straight + curvature_clipped * (v_curve - v_straight);
        return std::max(v_min, std::min(v_max, target_speed));
    }

    void load_waypoints(const std::string& filename) {
        std::vector<std::pair<double, double>> raw_waypoints;
        std::ifstream file(filename);
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string val;
            std::vector<std::string> tokens;
            while (std::getline(ss, val, ',')) {
                tokens.push_back(val);
            }
            if (tokens.size() < 2) {
                tokens.clear();
                std::stringstream ss2(line);
                while (ss2 >> val) tokens.push_back(val);
            }
            if (tokens.size() >= 2) {
                try {
                    raw_waypoints.push_back({std::stod(tokens[0]), std::stod(tokens[1])});
                } catch (...) {}
            }
        }
        
        auto smoothed = smooth_path(raw_waypoints);
        waypoints.clear();
        for (size_t i = 0; i < smoothed.size(); i++) {
            auto p1 = smoothed[i];
            auto p2 = smoothed[(i + 1) % smoothed.size()];
            double yaw = std::atan2(p2.second - p1.second, p2.first - p1.first);
            waypoints.push_back({p1.first, p1.second, yaw});
        }
    }

    std::vector<std::pair<double, double>> smooth_path(const std::vector<std::pair<double, double>>& path, 
                                                       double weight_data=0.5, double weight_smooth=0.2, double tolerance=0.00001) {
        auto new_path = path;
        double change = tolerance;
        while (change >= tolerance) {
            change = 0.0;
            for (size_t i = 1; i < path.size() - 1; i++) {
                double ax = new_path[i].first;
                double ay = new_path[i].second;
                new_path[i].first += (weight_data * (path[i].first - new_path[i].first) + 
                                      weight_smooth * (new_path[i-1].first + new_path[i+1].first - 2.0 * new_path[i].first));
                new_path[i].second += (weight_data * (path[i].second - new_path[i].second) + 
                                       weight_smooth * (new_path[i-1].second + new_path[i+1].second - 2.0 * new_path[i].second));
                change += std::abs(ax - new_path[i].first) + std::abs(ay - new_path[i].second);
            }
        }
        return new_path;
    }

    int find_nearest_waypoint(double rx, double ry) {
        if (start_index == -1) {
            double min_dist = 1e9;
            int idx = 0;
            for (size_t i = 0; i < waypoints.size(); i++) {
                double d = std::hypot(rx - waypoints[i].x, ry - waypoints[i].y);
                if (d < min_dist) {
                    min_dist = d;
                    idx = i;
                }
            }
            start_index = idx;
            return idx;
        }

        int idx = start_index;
        double curr_d = std::hypot(rx - waypoints[idx].x, ry - waypoints[idx].y);
        for (int i = 0; i < 20; i++) {
            int nxt = (idx + 1) % waypoints.size();
            double nxt_d = std::hypot(rx - waypoints[nxt].x, ry - waypoints[nxt].y);
            if (nxt_d < curr_d) {
                idx = nxt;
                curr_d = nxt_d;
            } else {
                break;
            }
        }
        start_index = idx;
        return idx;
    }

    void build_reference(int nearest_idx, double wp_x, double wp_y, double wp_yaw, double v_target,
                         Eigen::VectorXd& r_vector, std::vector<std::pair<double, double>>& ref_global_points) {
        r_vector = Eigen::VectorXd::Zero(2 * hz);
        double step_dist = std::max(v_x_current, 2.5) * Ts;
        int curr_idx = nearest_idx;
        double dist_accum = 0.0;

        for (int i = 1; i <= hz; i++) {
            double target_dist = i * step_dist;
            double fx = 0, fy = 0, fyaw = 0;

            while (true) {
                int next_idx = (curr_idx + 1) % waypoints.size();
                auto p1 = waypoints[curr_idx];
                auto p2 = waypoints[next_idx];
                double segment_len = std::hypot(p2.x - p1.x, p2.y - p1.y);

                if (dist_accum + segment_len >= target_dist) {
                    double ratio = segment_len > 0 ? (target_dist - dist_accum) / segment_len : 0.0;
                    fx = p1.x + ratio * (p2.x - p1.x);
                    fy = p1.y + ratio * (p2.y - p1.y);
                    fyaw = std::atan2(p2.y - p1.y, p2.x - p1.x);
                    break;
                } else {
                    dist_accum += segment_len;
                    curr_idx = next_idx;
                }
            }

            ref_global_points.push_back({fx, fy});
            double fdx = fx - wp_x;
            double fdy = fy - wp_y;
            double local_y = -std::sin(wp_yaw) * fdx + std::cos(wp_yaw) * fdy;
            double local_yaw = normalize_angle(fyaw - wp_yaw);

            r_vector(2 * (i - 1)) = local_yaw;
            r_vector(2 * (i - 1) + 1) = local_y;
        }
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        if (waypoints.empty()) return;

        auto current_time = this->get_clock()->now();
        if ((current_time - last_mpc_time).seconds() < Ts) return;
        last_mpc_time = current_time;

        double v_x = msg->twist.twist.linear.x;
        double v_y = msg->twist.twist.linear.y;
        double yaw_rate = msg->twist.twist.angular.z;
        v_x_current = std::max(v_x, v_min);

        double rx, ry, r_yaw;
        try {
            auto transform = tf_buffer->lookupTransform(map_frame, car_frame, tf2::TimePointZero);
            rx = transform.transform.translation.x;
            ry = transform.transform.translation.y;
            tf2::Quaternion q(
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w);
            tf2::Matrix3x3 m(q);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);
            r_yaw = yaw;
        } catch (tf2::TransformException &ex) {
            return;
        }

        int nearest_idx = find_nearest_waypoint(rx, ry);
        double wp_x = waypoints[nearest_idx].x;
        double wp_y = waypoints[nearest_idx].y;
        double wp_yaw = waypoints[nearest_idx].yaw;

        double e_psi = normalize_angle(r_yaw - wp_yaw);
        double dx = rx - wp_x;
        double dy = ry - wp_y;
        double e_y = -std::sin(wp_yaw) * dx + std::cos(wp_yaw) * dy;

        double v_target = compute_target_speed(nearest_idx);

        Eigen::MatrixXd Ad, Bd, Cd;
        calculate_state_space(v_x_current, Ad, Bd, Cd);

        Eigen::MatrixXd Hdb, Fdbt, Cdb, Adc;
        mpc_simplification(Ad, Bd, Cd, Hdb, Fdbt, Cdb, Adc);

        Eigen::VectorXd x_aug_t(7);
        x_aug_t << v_y, e_psi, yaw_rate, e_y, v_x_current, U1, U2;

        Eigen::VectorXd r_vector;
        std::vector<std::pair<double, double>> ref_global_points;
        build_reference(nearest_idx, wp_x, wp_y, wp_yaw, v_target, r_vector, ref_global_points);

        publish_mpc_reference(ref_global_points);

        Eigen::VectorXd ft_input(7 + 2 * hz);
        ft_input << x_aug_t, r_vector;
        Eigen::VectorXd ft = Fdbt.transpose() * ft_input;
        
        Eigen::MatrixXd Hdb_sym = 0.5 * (Hdb + Hdb.transpose());
        Hdb_sym += 1e-8 * Eigen::MatrixXd::Identity(Hdb.rows(), Hdb.cols());

        Eigen::MatrixXd G;
        Eigen::VectorXd ht;
        build_constraints(G, ht);

        // eiquadprog solves: min 0.5 * x G x + g0 x s.t. CI^T x + ci0 >= 0
        // We have: G * du <= ht  =>  -G * du + ht >= 0
        Eigen::MatrixXd CI = -G.transpose();
        Eigen::VectorXd ci0 = ht;
        Eigen::MatrixXd CE = Eigen::MatrixXd::Zero(Hdb.rows(), 0);
        Eigen::VectorXd ce0 = Eigen::VectorXd::Zero(0);
        Eigen::VectorXd du = Eigen::VectorXd::Zero(2 * hz);
        Eigen::VectorXi activeSet;
        size_t activeSetSize;

        double cost = eiquadprog::solvers::solve_quadprog(Hdb_sym, ft, CE, ce0, CI, ci0, du, activeSet, activeSetSize);
        if (cost == std::numeric_limits<double>::infinity()) {
            RCLCPP_WARN(this->get_logger(), "QP thất bại");
            // Unconstrained fallback
            du = -Hdb_sym.llt().solve(ft);
        }

        U1 = std::max(delta_min, std::min(delta_max, U1 + du(0)));
        U2 = std::max(a_min, std::min(a_max, U2 + du(1)));

        Eigen::VectorXd X_pred = Adc * x_aug_t + Cdb * du;
        std::vector<std::pair<double, double>> predict_global_points;
        double s_accum = 0.0;
        for (int i = 0; i < hz; i++) {
            double e_y_pred = X_pred(7 * i + 3);
            double v_x_pred = X_pred(7 * i + 4);
            s_accum += v_x_pred * Ts;
            double px = wp_x + s_accum * std::cos(wp_yaw) - e_y_pred * std::sin(wp_yaw);
            double py = wp_y + s_accum * std::sin(wp_yaw) + e_y_pred * std::cos(wp_yaw);
            predict_global_points.push_back({px, py});
        }
        publish_mpc_predict(predict_global_points, rx, ry);

        v_x_current = std::max(v_min, std::min(v_max, v_x_current + U2 * Ts));

        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.drive.steering_angle = U1;
        drive_msg.drive.speed = v_target;
        pub_drive->publish(drive_msg);
        
        RCLCPP_INFO(this->get_logger(), "toc do %f", drive_msg.drive.speed);
    }

    void publish_mpc_reference(const std::vector<std::pair<double, double>>& points_list) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = map_frame;
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "mpc_reference";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.15;
        marker.scale.y = 0.15;
        marker.scale.z = 0.15;
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;
        for (auto& pt : points_list) {
            geometry_msgs::msg::Point p;
            p.x = pt.first;
            p.y = pt.second;
            p.z = 0.1;
            marker.points.push_back(p);
        }
        pub_mpc_ref->publish(marker);
    }

    void publish_full_waypoint() {
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->get_clock()->now();
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.05;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        for (auto& wp : waypoints) {
            geometry_msgs::msg::Point p;
            p.x = wp.x;
            p.y = wp.y;
            p.z = 0.0;
            marker.points.push_back(p);
        }
        if (!waypoints.empty()) {
            geometry_msgs::msg::Point p;
            p.x = waypoints[0].x;
            p.y = waypoints[0].y;
            p.z = 0.0;
            marker.points.push_back(p);
        }
        marker_array.markers.push_back(marker);
        pub_marker_path->publish(marker_array);
    }

    void publish_mpc_predict(const std::vector<std::pair<double, double>>& points_list, double current_x, double current_y) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = map_frame;
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "mpc_predict";
        marker.id = 1;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.08;
        marker.color.a = 0.8;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        
        geometry_msgs::msg::Point p_start;
        p_start.x = current_x;
        p_start.y = current_y;
        p_start.z = 0.15;
        marker.points.push_back(p_start);

        for (auto& pt : points_list) {
            geometry_msgs::msg::Point p;
            p.x = pt.first;
            p.y = pt.second;
            p.z = 0.15;
            marker.points.push_back(p);
        }
        pub_mpc_predict->publish(marker);
    }
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MPCNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
