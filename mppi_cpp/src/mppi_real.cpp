/**
 * mppi_real.cpp
 * MPPI (Model Predictive Path Integral) Controller — Xe thật F1TENTH
 *
 * Kiến trúc tổng quan:
 *   ┌─────────────────────────────────────────────────────────┐
 *   │  lidar_callback  → map_obstacles (mutex)               │
 *   │  odom_callback   → pose + v_cur  (mutex)               │
 *   │  control_loop    → MPPI → publish /drive (20 Hz)       │
 *   └─────────────────────────────────────────────────────────┘
 *
 * Luồng xử lý trong control_loop:
 *   1. Snapshot pose + obstacles (thread-safe)
 *   2. Tìm waypoint gần nhất
 *   3. Corridor Filter: phân loại LiDAR → tường / vật cản
 *   4. Curvature Profiling → tốc độ mục tiêu
 *   5. Proactive Deceleration theo vật cản phía trước
 *   6. Kiểm tra front_blocked + anti-stuck watchdog
 *   7. Emergency Stop & Escape Maneuver
 *   8. MPPI sampling + tính cost song song (OpenMP)
 *   9. MPPI weight update → cập nhật nominal_control
 *  10. EMA smoothing → publish lệnh điều khiển
 */

#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <mutex>
#include <omp.h>

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

// ============================================================
// Struct hỗ trợ
// ============================================================
struct Point2D { double x, y; };
struct Control { double v, steer; };
struct ObsPt   { double x, y, r; };  // điểm chướng ngại vật với bán kính nguy hiểm

// ============================================================
// MPPIController Node
// ============================================================
class MPPIController : public rclcpp::Node {
public:
    MPPIController() : Node("mppi_real_controller_node") {

        // ROS Parameters (có thể override từ launch file / command line)
        declare_parameter("horizon",     30);
        declare_parameter("num_samples", 500);
        declare_parameter("dt",          0.05);
        horizon     = get_parameter("horizon").as_int();
        num_samples = get_parameter("num_samples").as_int();
        dt          = get_parameter("dt").as_double();

        // Pre-allocate MPPI buffers (tránh alloc trong vòng lặp)
        noise_buf.resize(num_samples, std::vector<Control>(horizon));
        costs_buf.resize(num_samples, 0.0);
        weights_buf.resize(num_samples, 0.0);
        nominal_control.resize(horizon, {0.0, 0.0});
        local_wps.reserve(100);
        local_hdgs.reserve(100);
        local_idxs.reserve(100);

        // TF
        tf_buffer   = std::make_unique<tf2_ros::Buffer>(get_clock());
        tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

        // Subscriptions
        sub_odom  = create_subscription<nav_msgs::msg::Odometry>(
            "/pf/pose/odom", 10,
            std::bind(&MPPIController::odom_callback, this, std::placeholders::_1));
        sub_laser = create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&MPPIController::lidar_callback, this, std::placeholders::_1));

        // Publishers
        pub_drive     = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        pub_best_traj = create_publisher<visualization_msgs::msg::Marker>("/mppi_best_trajectory", 10);
        pub_waypoints = create_publisher<visualization_msgs::msg::MarkerArray>("/publish_full_waypoint", 10);

        // Timer điều khiển (20 Hz khi dt=0.05)
        control_timer = create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(dt * 1000)),
            std::bind(&MPPIController::control_loop, this));

        rng = std::mt19937(std::random_device{}());

        load_waypoints(WAYPOINT_CSV_PATH);
        publish_waypoints_marker();

        RCLCPP_INFO(get_logger(),
            "MPPI Real Controller started. WPs=%zu, H=%d, N=%d, dt=%.3fs",
            waypoints.size(), horizon, num_samples, dt);
    }

private:
    // ============================================================
    // [A] THAM SỐ PHẦN CỨNG & CẤU HÌNH
    // ============================================================

    // Đường dẫn file waypoint CSV (x,y mỗi dòng)
    const std::string WAYPOINT_CSV_PATH =
        "/home/fablab_01/danh_pp_ws/install/waypoint/share/waypoint/"
        "f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv";

    // Thông số cơ học xe
    static constexpr double WHEELBASE     = 0.33;   // Chiều dài cơ sở (m)
    static constexpr double MAX_STEER_RAD = 0.418;  // Góc lái vật lý tối đa (rad)

    // Tên frame ROS
    const std::string car_frame = "base_link";
    const std::string map_frame = "map";

    // ============================================================
    // [B] THAM SỐ ĐIỀU CHỈNH (TUNE)
    // ============================================================

    // -- MPPI Cost Weights --
    double lambda_    = 120.0;  // Softmax temperature: cao→đều, thấp→tham lam trajectory tốt nhất
    double w_track    = 6.0;    // Bám tâm đường (Đã giảm từ 10 xuống 6 để xe dám lấn làn né vật)
    double w_heading  = 15.0;   // Song song đường đua (Đã giảm từ 35 xuống 15 để xe dám bẻ chéo đầu lách)
    double w_progress =  5.0;   // Khuyến khích tiến về phía trước
    double w_obs      = 150.0;  // Né chướng ngại vật (Tăng nhẹ để ưu tiên tính mạng)
    double w_smooth   = 10.5;   // Phạt bẻ lái/thay đổi tốc độ đột ngột (giảm rack-rack)
    double w_speed    =  8.0;   // Bám tốc độ mục tiêu

    // -- Tốc độ & Gia tốc --
    double target_speed_max = 2.5;  // Tốc độ tối đa (m/s)          [TUNE]
    double min_speed_curve  = 1.5;  // Tốc độ tối thiểu trong cua (m/s)
    double max_accel        = 2.5;  // Gia tốc tăng tốc tối đa (m/s²)
    double max_decel        = 4.0;  // Gia tốc phanh tối đa (m/s²)

    // -- Phát hiện cua (Curvature Profiling) --
    double curve_thresh        = 0.3;  // Độ cong ngưỡng cua gắt (1/m) [TUNE]
    int    speed_lookahead_wps = 25;   // Số waypoint nhìn trước để phát hiện cua

    // -- Corridor Filter (phân loại LiDAR → tường / vật cản) --
    //
    //   Cách hoạt động: transform điểm LiDAR sang car frame,
    //   nếu nằm trong hành lang phía trước → vật cản nguy hiểm,
    //   còn lại (2 bên, phía sau) → tường bình thường.
    //
    //   Ưu điểm so với Wall Filter cũ: không cần x0/y0 chính xác
    //   từ Particle Filter, chỉ cần heading theta0.
    //
    double corridor_half_w = 0.85;  // Nửa chiều rộng hành lang (m) - Đã tăng lên 0.85 để dễ lách
    double corridor_max_d  = 6.0;   // Chiều dài hành lang nhìn trước (m)
    double r_obstacle      = 0.30;  // Bán kính nguy hiểm (m) - GIẢM XUỐNG 0.30 (bằng đúng bề ngang xe). Nếu to quá xe sẽ sợ không dám lách.
    double r_wall          = 0.25;  // Bán kính nguy hiểm của tường 2 bên (m)
    double collision_cost  = 200.0; // Phạt cực nặng nếu quẹt trúng vật cản

    // -- Proactive Deceleration (giảm tốc sớm khi có vật cản) --
    double obs_decel_start_dist = 4.0;   // Bắt đầu giảm tốc khi vật cản cách (m)
    double obs_decel_min_factor = 0.25;  // Hệ số tốc độ tối thiểu khi vật cản rất gần

    // -- Emergency Stop & Recovery --
    double stuck_timer_thresh         = 1.0;  // Thời gian xác nhận bị kẹt (s)
    double stop_timer_duration        = 1.0;  // Thời gian thực hiện escape (s)
    double escape_speed               = 0.0;  // Tốc độ khi escape (m/s) - Đã đổi thành 0.0 (phanh đứng hình)
    double watchdog_cooldown_duration = 2.0;  // Cooldown watchdog sau escape (s)
    double front_blocked_cooldown_dur = 1.5;  // Cooldown front_blocked sau escape (s)

    // -- EMA Output Smoothing (giảm nhiễu MPPI → bớt rack-rack) --
    double alpha_v = 0.15;  // Hệ số EMA tốc độ:    nhỏ → mượt hơn, trễ hơn
    double alpha_s = 0.30;  // Hệ số EMA góc lái:   nhỏ → mượt hơn, trễ hơn

    // -- MPPI Exploration Noise --
    double sigma_v = 1.5;   // Độ lệch chuẩn nhiễu tốc độ MPPI
    double sigma_s = 0.30;  // Độ lệch chuẩn góc lái - Đã tăng lên 0.30 để xe có thể bẻ gắt né vật cản

    // -- MPPI Horizon (set từ ROS param) --
    int    horizon, num_samples;
    double dt;

    // ============================================================
    // [C] DỮ LIỆU WAYPOINT
    // ============================================================

    std::vector<Point2D> waypoints;           // Tất cả waypoints từ CSV
    std::vector<double>  waypoint_headings;   // Heading tại mỗi waypoint (atan2)
    std::vector<double>  waypoint_curvatures; // Độ cong tại mỗi waypoint

    // Cửa sổ waypoint cục bộ (được tính lại mỗi vòng lặp)
    std::vector<Point2D> local_wps;
    std::vector<double>  local_hdgs;
    std::vector<int>     local_idxs;

    // ============================================================
    // [D] BUFFER MPPI (pre-allocated)
    // ============================================================

    std::vector<Control>              nominal_control; // Quỹ đạo điều khiển danh nghĩa
    std::vector<std::vector<Control>> noise_buf;       // Nhiễu ngẫu nhiên [N × H]
    std::vector<double>               costs_buf;       // Chi phí mỗi mẫu
    std::vector<double>               weights_buf;     // Trọng số softmax mỗi mẫu

    // ============================================================
    // [E] TRẠNG THÁI RUNTIME
    // ============================================================

    // Pose + vận tốc (được bảo vệ bởi pose_mutex — đọc/ghi từ 2 thread khác nhau)
    double       x0 = 0.0, y0 = 0.0, theta0 = 0.0, v_cur = 0.0;
    bool         odom_received = false;
    rclcpp::Time odom_stamp;
    std::mutex   pose_mutex;

    // LiDAR obstacles trong map frame (được bảo vệ bởi obs_mutex)
    std::vector<Point2D> map_obstacles;
    rclcpp::Time         obstacle_stamp;
    std::mutex           obs_mutex;

    // Vị trí waypoint gần nhất (cache giữa các vòng lặp để tăng tốc tìm kiếm)
    int last_nearest_wp = 0;

    // Trạng thái bộ lọc tốc độ & góc lái (EMA)
    double last_target_speed = 0.0;
    double last_ema_v        = 0.0;
    double last_ema_steer    = 0.0;

    // Trạng thái Emergency Stop
    bool   is_stopped                   = false;
    bool   is_stuck_timer_active        = false;
    double stuck_start_time             = 0.0;
    double stop_end_time                = 0.0;
    double watchdog_cooldown_until      = 0.0;
    double front_blocked_cooldown_until = 0.0;

    std::mt19937 rng;

    // ============================================================
    // [F] ROS INTERFACES
    // ============================================================

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr     sub_odom;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_laser;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr pub_drive;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr            pub_best_traj;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr       pub_waypoints;
    rclcpp::TimerBase::SharedPtr control_timer;

    std::unique_ptr<tf2_ros::Buffer>            tf_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener;

    // ============================================================
    // [1] KHỞI TẠO — Đọc waypoints từ CSV, tính heading & curvature
    // ============================================================

    void load_waypoints(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            RCLCPP_FATAL(get_logger(), "Không mở được file waypoint: %s", path.c_str());
            rclcpp::shutdown();
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            std::string v1, v2;
            if (std::getline(ss, v1, ',') && std::getline(ss, v2, ',')) {
                try { waypoints.push_back({std::stod(v1), std::stod(v2)}); }
                catch (...) { continue; }
            }
        }
        if (waypoints.empty()) {
            RCLCPP_FATAL(get_logger(), "File waypoint rỗng!");
            rclcpp::shutdown();
            return;
        }

        // Tính heading và curvature bằng 5-point span (mượt, ít nhạy với nhiễu)
        int w = static_cast<int>(waypoints.size());
        waypoint_headings.resize(w);
        waypoint_curvatures.resize(w);
        for (int i = 0; i < w; i++) {
            auto& p1 = waypoints[(i - 5 + w) % w];
            auto& p2 = waypoints[i];
            auto& p3 = waypoints[(i + 5) % w];
            waypoint_headings[i] = std::atan2(p3.y - p1.y, p3.x - p1.x);
            double dx1 = p2.x - p1.x, dy1 = p2.y - p1.y;
            double dx2 = p3.x - p2.x, dy2 = p3.y - p2.y;
            double l1 = std::hypot(dx1, dy1), l2 = std::hypot(dx2, dy2);
            double l3 = std::hypot(p3.x - p1.x, p3.y - p1.y);
            waypoint_curvatures[i] = (l1 * l2 * l3 > 1e-9)
                ? 4.0 * (dx1 * dy2 - dy1 * dx2) / (l1 * l2 * l3)
                : 0.0;
        }
        RCLCPP_INFO(get_logger(), "Đã tải %d waypoints.", w);
    }

    // ============================================================
    // [2] CALLBACK — Odometry: nhận pose + vận tốc từ Particle Filter
    // ============================================================

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        auto& q  = msg->pose.pose.orientation;
        double siny = 2.0 * (q.w * q.z + q.x * q.y);
        double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);

        // Dùng mutex vì control_loop (thread khác) đọc các biến này
        std::lock_guard<std::mutex> lock(pose_mutex);
        x0     = msg->pose.pose.position.x;
        y0     = msg->pose.pose.position.y;
        theta0 = std::atan2(siny, cosy);
        v_cur  = msg->twist.twist.linear.x;
        odom_received = true;
        odom_stamp    = now();
    }

    // ============================================================
    // [3] CALLBACK — LiDAR: chuyển điểm scan sang map frame
    // ============================================================

    void lidar_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        // Lấy transform từ laser frame → map frame
        geometry_msgs::msg::TransformStamped tf;
        try {
            tf = tf_buffer->lookupTransform(map_frame, msg->header.frame_id, tf2::TimePointZero);
        } catch (...) { return; }

        auto& q = tf.transform.rotation;
        double yaw = std::atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z));
        double tx  = tf.transform.translation.x;
        double ty  = tf.transform.translation.y;

        // Down-sample theo góc 1° để giảm khối lượng tính toán
        int step = std::max(1, static_cast<int>(M_PI / 180.0 / msg->angle_increment));

        std::vector<Point2D> temp;
        temp.reserve(msg->ranges.size() / step + 1);
        for (size_t i = 0; i < msg->ranges.size(); i += step) {
            double r = msg->ranges[i];
            if (!std::isnormal(r) || r < 0.1 || r > 3.5) continue;
            double angle = msg->angle_min + i * msg->angle_increment;
            double px = r * std::cos(angle);
            double py = r * std::sin(angle);
            temp.push_back({
                tx + px * std::cos(yaw) - py * std::sin(yaw),
                ty + px * std::sin(yaw) + py * std::cos(yaw)
            });
        }

        // Swap O(1) thay vì copy O(n)
        std::lock_guard<std::mutex> lock(obs_mutex);
        std::swap(map_obstacles, temp);
        obstacle_stamp = now();
    }

    // ============================================================
    // [4] VÒNG LẶP ĐIỀU KHIỂN CHÍNH (20 Hz)
    // ============================================================

    void control_loop() {
        double now_s = now().seconds();

        // --- 4.1 Snapshot thread-safe ---
        // Sao chép pose + obstacles vào biến local để tránh race condition
        // trong suốt phần còn lại của vòng lặp.
        double x, y, th, vc;
        bool   got_odom;
        rclcpp::Time o_stamp;
        {
            std::lock_guard<std::mutex> lock(pose_mutex);
            x = x0; y = y0; th = theta0; vc = v_cur;
            got_odom = odom_received;
            o_stamp  = odom_stamp;
        }
        std::vector<Point2D> raw_obs;
        rclcpp::Time         obs_stamp;
        {
            std::lock_guard<std::mutex> lock(obs_mutex);
            raw_obs   = map_obstacles;
            obs_stamp = obstacle_stamp;
        }

        // --- 4.2 Kiểm tra tính hợp lệ của dữ liệu đầu vào ---
        if (!got_odom || waypoints.empty() || (now_s - o_stamp.seconds() > 0.5)) {
            publish_drive(0.0, 0.0); // Dừng an toàn nếu mất tín hiệu
            return;
        }
        if (std::isnan(x) || std::isnan(y) || std::isnan(th)) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "NaN trong pose!");
            return;
        }

        // --- 4.3 Tìm waypoint gần nhất ---
        // Tìm trong cửa sổ [-20, +40] quanh vị trí cuối để tăng tốc (O(60)).
        // Nếu xe bị dịch chuyển xa > 5m (teleport), tìm toàn bộ (O(N)).
        int nearest_wp = last_nearest_wp;
        {
            double min_d = 9999.0;
            for (int di = -20; di <= 40; di++) {
                int idx = (nearest_wp + di + (int)waypoints.size()) % (int)waypoints.size();
                double d = std::hypot(waypoints[idx].x - x, waypoints[idx].y - y);
                if (d < min_d) { min_d = d; nearest_wp = idx; }
            }
            if (min_d > 5.0) { // Teleport recovery
                for (int i = 0; i < (int)waypoints.size(); i++) {
                    double d = std::hypot(waypoints[i].x - x, waypoints[i].y - y);
                    if (d < min_d) { min_d = d; nearest_wp = i; }
                }
            }
            last_nearest_wp = nearest_wp;
        }

        // --- 4.4 Xây dựng cửa sổ waypoint cục bộ ---
        // Lấy 80 waypoint xung quanh nearest_wp (-15 đến +65) để MPPI tìm kiếm.
        const int WP_WINDOW = 80;
        local_wps.clear(); local_hdgs.clear(); local_idxs.clear();
        for (int i = -15; i < WP_WINDOW - 15; i++) {
            int idx = ((nearest_wp + i) % (int)waypoints.size() + (int)waypoints.size()) % (int)waypoints.size();
            local_wps.push_back(waypoints[idx]);
            local_hdgs.push_back(waypoint_headings[idx]);
            local_idxs.push_back(idx);
        }
        int local_nearest = -1;
        for (int i = 0; i < (int)local_idxs.size(); i++) {
            if (local_idxs[i] == nearest_wp) { local_nearest = i; break; }
        }
        if (local_nearest == -1) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "local_nearest không tìm thấy!");
            return;
        }

        // --- 4.5 Corridor Filter: phân loại LiDAR → tường / vật cản ---
        //
        // Mỗi điểm LiDAR được transform sang car frame (hệ tọa độ gắn với xe).
        //  • Nằm trong hành lang phía trước (|dy| < corridor_half_w, dx > 0.1):
        //    → VẬT CẢN, r_danger = r_obstacle (lớn, MPPI né từ xa)
        //  • Còn lại (2 bên, phía sau):
        //    → TƯỜNG, r_danger = r_wall (nhỏ, xe được phép chạy gần)
        //
        // Không dùng raceline để phân loại → không phụ thuộc localization accuracy.
        //
        std::vector<ObsPt> obs_pts;
        obs_pts.reserve(raw_obs.size());
        int    wall_cnt = 0, obs_cnt = 0;
        double min_front_dist = 999.0;

        for (const auto& pt : raw_obs) {
            double dx   = pt.x - x, dy = pt.y - y;
            double dx_l = dx * std::cos(-th) - dy * std::sin(-th); // trục X xe (phía trước)
            double dy_l = dx * std::sin(-th) + dy * std::cos(-th); // trục Y xe (bên trái)

            double r;
            if (dx_l > 0.1 && dx_l < corridor_max_d && std::abs(dy_l) < corridor_half_w) {
                r = r_obstacle;
                obs_cnt++;
                if (dx_l < min_front_dist) min_front_dist = dx_l;
            } else {
                r = r_wall;
                wall_cnt++;
            }
            obs_pts.push_back({pt.x, pt.y, r});
        }
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
            "Corridor: wall=%d(r=%.2f) | obs=%d(r=%.2f) | front=%.2fm",
            wall_cnt, r_wall, obs_cnt, r_obstacle, min_front_dist);

        // --- 4.6 Curvature Profiling → tốc độ mục tiêu theo hình dạng đường ---
        // Nhìn trước speed_lookahead_wps waypoints để phát hiện cua và giảm tốc sớm.
        double max_c = 0.0;
        for (int i = 0; i < speed_lookahead_wps; i++) {
            double c = std::abs(waypoint_curvatures[(nearest_wp + i) % (int)waypoints.size()]);
            if (c > max_c) max_c = c;
        }
        // speed_factor: 1.0 khi thẳng, giảm dần khi cua gắt
        double speed_factor = (max_c > curve_thresh)
            ? std::max(0.0, 1.0 - (max_c - curve_thresh) / curve_thresh)
            : 1.0;
        double target_v = min_speed_curve + (target_speed_max - min_speed_curve) * speed_factor;

        // --- 4.7 Proactive Deceleration: giảm tốc sớm khi vật cản tiến gần ---
        if (min_front_dist < obs_decel_start_dist) {
            double f = obs_decel_min_factor
                + (1.0 - obs_decel_min_factor) * ((min_front_dist - 0.5) / (obs_decel_start_dist - 0.5));
            f = std::max(obs_decel_min_factor, std::min(1.0, f));
            target_v = std::min(target_v, target_speed_max * f);
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 200,
                "Proactive Decel: dist=%.2fm → tgt_v=%.2fm/s", min_front_dist, target_v);
        }

        // Giới hạn tốc độ thay đổi theo gia tốc / phanh tối đa (tránh jump đột ngột)
        target_v = std::min(last_target_speed + max_accel * dt, target_v);
        target_v = std::max(last_target_speed - max_decel * dt, target_v);
        last_target_speed = target_v;

        // --- 4.8 Kiểm tra front_blocked: vật cản trong khoảng phanh ---
        // Chỉ kiểm tra khi dữ liệu LiDAR còn mới (<0.5s) và hết cooldown sau escape.
        bool   front_blocked = false;
        double sum_dy        = 0.0; // Tổng dy vật cản (để chọn hướng thoát)
        int    blk_cnt       = 0;

        double obs_age = now_s - obs_stamp.seconds();
        bool obs_fresh = (obs_stamp.nanoseconds() != 0 && obs_age < 0.5);
        
        // Log lỗi 1: Cảnh báo nếu LiDAR bị trễ (vô hiệu hóa phanh khẩn cấp)
        if (obs_stamp.nanoseconds() != 0 && !obs_fresh) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, 
                "[CẢNH BÁO] LiDAR bị trễ %.2fs (>0.5s). Phanh khẩn cấp đang bị TẮT để tránh lỗi!", obs_age);
        }

        if (obs_fresh && now_s > front_blocked_cooldown_until) {
            // Khoảng cách phanh = v²/(2a) + margin
            double braking_d = std::max(0.8, vc * vc / (2.0 * max_decel) + 0.3);
            double pre_bound = braking_d + 0.5; // Pre-filter nhanh (AABB)

            for (const auto& o : obs_pts) {
                double dx = o.x - x, dy = o.y - y;
                if (std::abs(dx) > pre_bound || std::abs(dy) > pre_bound) continue;
                double dx_l = dx * std::cos(-th) - dy * std::sin(-th);
                double dy_l = dx * std::sin(-th) + dy * std::cos(-th);
                if (dx_l > 0.1 && dx_l < braking_d && std::abs(dy_l) < 0.35) {
                    sum_dy += dy_l;
                    blk_cnt++;
                    // Log điểm rơi vào vùng tử thần (để check lỗi TF Ghosting hoặc nhiễu)
                    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500, 
                        "[DEBUG LiDAR] Tia chạm vùng nguy hiểm: dx_l=%.2f (cần <%.2f), dy_l=%.2f (cần <0.35)", dx_l, braking_d, dy_l);
                }
            }
            
            // Sửa lỗi 2: Chống nhiễu LiDAR (Bóng ma) - Yêu cầu ít nhất 3 điểm
            if (blk_cnt >= 3) {
                front_blocked = true;
                RCLCPP_WARN(get_logger(), "[PHANH KHẨN CẤP] Phát hiện vật cản thật! Số tia chạm: %d, Tổng lệch ngang(sum_dy): %.2f", blk_cnt, sum_dy);
            } else if (blk_cnt > 0) {
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500, "[BỎ QUA] Có %d tia chạm vật cản, nhưng < 3 tia (Nhiễu/Bóng ma).", blk_cnt);
            }
        }

        // --- 4.9 Anti-stuck Watchdog ---
        // Phát hiện khi xe không di chuyển được dù MPPI ra lệnh chạy.
        // Chờ stuck_timer_thresh giây trước khi trigger để tránh false positive.
        bool is_stuck = false;
        if (!is_stopped && now_s > watchdog_cooldown_until
            && vc < 0.05 && std::abs(nominal_control[0].v) > 0.3)
        {
            if (!is_stuck_timer_active) {
                RCLCPP_WARN(get_logger(),
                    "[WATCHDOG BẮT ĐẦU ĐẾM] VESC báo xe đứng yên (v_cur=%.2f) nhưng MPPI bắt chạy (cmd_v=%.2f).", vc, nominal_control[0].v);
                stuck_start_time      = now_s;
                is_stuck_timer_active = true;
            } else if (now_s - stuck_start_time > stuck_timer_thresh) {
                is_stuck              = true;
                is_stuck_timer_active = false;
                RCLCPP_WARN(get_logger(), "[WATCHDOG KÍCH HOẠT] Đã quá %.1fs. Xác nhận xe bị kẹt!", stuck_timer_thresh);
            }
        } else if (!is_stopped && vc > 0.1) {
            if (is_stuck_timer_active) {
                RCLCPP_INFO(get_logger(), "[WATCHDOG HỦY] Xe đã trôi lại (v_cur=%.2f).", vc);
            }
            is_stuck_timer_active = false; // Reset timer khi xe đang chạy
        }

        // --- 4.10 Emergency Stop & Escape Maneuver ---
        // Khi phát hiện bị chặn hoặc bị kẹt: flush toàn bộ horizon về lệnh thoát,
        // sau đó đợi stop_timer_duration giây rồi resume.
        if (!is_stopped && (front_blocked || is_stuck)) {
            RCLCPP_WARN(get_logger(),
                "[TRẠNG THÁI] VÀO CHẾ ĐỘ ESCAPE (LÙI/ĐÁNH LÁI KHẨN)! Lý do: front_blocked=%d, is_stuck=%d", front_blocked, is_stuck);
            is_stopped    = true;
            stop_end_time = now_s + stop_timer_duration;
            last_ema_v    = 0.0; // Reset EMA để phản ứng ngay
            last_ema_steer = 0.0;

            // Hướng thoát: né về phía ít vật cản hơn
            double esc_steer = (blk_cnt >= 3)
                ? ((sum_dy > 0) ? -MAX_STEER_RAD : MAX_STEER_RAD)
                : ((rng() % 2 == 0) ? MAX_STEER_RAD : -MAX_STEER_RAD);

            for (auto& c : nominal_control) { c.v = escape_speed; c.steer = esc_steer; }
        }

        // Kết thúc escape → resume bình thường + áp dụng cooldown
        if (is_stopped && now_s > stop_end_time) {
            RCLCPP_INFO(get_logger(), "ESCAPE HOÀN THÀNH. Tiếp tục chạy.");
            is_stopped                   = false;
            is_stuck_timer_active        = false;
            watchdog_cooldown_until      = now_s + watchdog_cooldown_duration;
            front_blocked_cooldown_until = now_s + front_blocked_cooldown_dur;
            for (auto& c : nominal_control) { c.v = target_v; c.steer = 0.0; }
        }

        // Giới hạn vận tốc MPPI sampling: [0, target_v] khi bình thường, [0, escape_speed] khi stop
        double v_min = 0.0;
        double v_max = is_stopped ? escape_speed : target_v;
        if (is_stopped) target_v = escape_speed;
        double w_prog_eff = is_stopped ? 0.0 : w_progress; // Không khuyến khích tiến khi đang escape

        // --- 4.11 MPPI: Sinh mẫu ngẫu nhiên ---
        std::normal_distribution<double> dist_v(0.0, sigma_v);
        std::normal_distribution<double> dist_s(0.0, sigma_s);
        for (int n = 0; n < num_samples; n++)
            for (int t = 0; t < horizon; t++) {
                noise_buf[n][t].v     = dist_v(rng);
                noise_buf[n][t].steer = dist_s(rng);
            }

        // --- 4.12 MPPI: Tính chi phí song song (OpenMP) ---
        // Mỗi thread xử lý một trajectory mẫu độc lập.
        // Viết vào costs_buf[n] → không cần mutex (mỗi n có slot riêng).
        #pragma omp parallel for schedule(dynamic)
        for (int n = 0; n < num_samples; n++) {
            double px = x, py = y, pth = th;
            double trk = 0.0, hdg = 0.0, spd = 0.0, smo = 0.0, obs = 0.0;
            int    g_idx = nearest_wp;
            double pv_prv = nominal_control[0].v;
            double ps_prv = nominal_control[0].steer;
            double terminal = 0.0;

            for (int t = 0; t < horizon; t++) {
                // Điều khiển có nhiễu (clamp vào giới hạn vật lý)
                double pv = std::max(v_min, std::min(v_max,
                    nominal_control[t].v + noise_buf[n][t].v));
                double ps = std::max(-MAX_STEER_RAD, std::min(MAX_STEER_RAD,
                    nominal_control[t].steer + noise_buf[n][t].steer));

                // Mô hình xe đạp (bicycle model)
                px  += pv * std::cos(pth) * dt;
                py  += pv * std::sin(pth) * dt;
                pth += pv * std::tan(ps) / WHEELBASE * dt;
                pth  = normalize_angle(pth);

                // Waypoint gần nhất → cross-track + heading error
                double min_d2 = 999.0; int min_wi = 0;
                for (int wi = 0; wi < (int)local_wps.size(); wi++) {
                    double d2 = (px - local_wps[wi].x) * (px - local_wps[wi].x)
                              + (py - local_wps[wi].y) * (py - local_wps[wi].y);
                    if (d2 < min_d2) { min_d2 = d2; min_wi = wi; }
                }
                trk   += min_d2;
                g_idx  = local_idxs[min_wi];

                double herr = normalize_angle(pth - local_hdgs[min_wi]);
                hdg += herr * herr;

                // Tốc độ & smoothness
                spd += (pv - target_v) * (pv - target_v);
                if (t > 0) smo += (ps - ps_prv) * (ps - ps_prv) + (pv - pv_prv) * (pv - pv_prv);
                pv_prv = pv; ps_prv = ps;

                // Chi phí chướng ngại vật: phạt theo (r - d)² khi trong vùng nguy hiểm
                double max_pen = 0.0, min_abs_d = 999.0;
                for (const auto& o : obs_pts) {
                    double odx = px - o.x; if (std::abs(odx) > o.r) continue;
                    double ody = py - o.y; if (std::abs(ody) > o.r) continue;
                    double d   = std::hypot(odx, ody);
                    if (d < min_abs_d) min_abs_d = d;
                    if (d < o.r) { double p = (o.r - d) * (o.r - d); if (p > max_pen) max_pen = p; }
                }
                obs += max_pen;
                if (min_abs_d < 0.2) obs += collision_cost; // Va chạm trực tiếp

                // Chi phí terminal (bước cuối): bám vị trí + tiến về phía trước
                if (t == horizon - 1) {
                    double wh = is_stopped ? 5.0 : w_heading;
                    terminal += 3.0 * w_track * min_d2 + 3.0 * wh * herr * herr;
                    int prog = (g_idx - local_idxs[local_nearest] + (int)waypoints.size())
                               % (int)waypoints.size();
                    double prog_v = (prog > (int)waypoints.size() / 2)
                        ? (double)(waypoints.size() - prog)
                        : -(double)prog;
                    terminal += w_prog_eff * prog_v;
                }
            }
            costs_buf[n] = terminal
                + w_track * trk + w_heading * hdg + w_speed * spd
                + w_smooth * smo + w_obs * obs;
        }

        // --- 4.13 MPPI: Tính trọng số & cập nhật nominal_control ---
        double min_cost = *std::min_element(costs_buf.begin(), costs_buf.end());
        double w_sum    = 0.0;
        for (int n = 0; n < num_samples; n++) {
            weights_buf[n] = std::exp(-(costs_buf[n] - min_cost) / lambda_);
            w_sum += weights_buf[n];
        }

        if (w_sum > 1e-10) {
            // Tích lũy update (loop n trước, t sau để cache-friendly)
            std::vector<double> upd_v(horizon, 0.0), upd_s(horizon, 0.0);
            for (int n = 0; n < num_samples; n++) {
                double w = weights_buf[n];
                for (int t = 0; t < horizon; t++) {
                    upd_v[t] += w * noise_buf[n][t].v;
                    upd_s[t] += w * noise_buf[n][t].steer;
                }
            }
            for (int t = 0; t < horizon; t++) {
                nominal_control[t].v     = std::max(v_min, std::min(v_max,
                    nominal_control[t].v + upd_v[t] / w_sum));
                nominal_control[t].steer = std::max(-MAX_STEER_RAD, std::min(MAX_STEER_RAD,
                    nominal_control[t].steer + upd_s[t] / w_sum));
            }
        } else {
            // Weight collapse: tất cả trajectory đều xấu, giữ nguyên nominal
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPPI weight collapse (w_sum≈0)! Giữ nguyên nominal control.");
            for (int t = 0; t < horizon; t++) {
                nominal_control[t].v     = std::max(v_min, std::min(v_max, nominal_control[t].v));
                nominal_control[t].steer = std::max(-MAX_STEER_RAD, std::min(MAX_STEER_RAD, nominal_control[t].steer));
            }
        }

        // --- 4.14 EMA Smoothing: giảm nhiễu MPPI trước khi publish ---
        // MPPI output thường dao động nhanh giữa các step. EMA làm mượt output
        // mà không thay đổi behavior tổng thể.
        last_ema_v     = alpha_v * nominal_control[0].v     + (1.0 - alpha_v) * last_ema_v;
        last_ema_steer = alpha_s * nominal_control[0].steer + (1.0 - alpha_s) * last_ema_steer;

        publish_drive(last_ema_v, last_ema_steer);
        publish_best_trajectory(x, y, th);

        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 100,
            "MPPI | cost=%6.0f | curv=%.2f | tgt=%.2f | v=%.2f | ema_v=%.2f | ema_s=%5.2f | blk=%d stk=%d stp=%d",
            min_cost, max_c, target_v, vc, last_ema_v, last_ema_steer, front_blocked, is_stuck, is_stopped);

        // --- 4.15 Shift horizon: chuẩn bị cho vòng lặp tiếp theo ---
        for (int t = 0; t < horizon - 1; t++) nominal_control[t] = nominal_control[t + 1];
        nominal_control[horizon - 1].v     = target_v;
        nominal_control[horizon - 1].steer = nominal_control[horizon - 2].steer * 0.5; // Suy giảm góc lái cuối
    }

    // ============================================================
    // [5] HÀM PUBLISH
    // ============================================================

    void publish_drive(double v, double steer) {
        if (std::isnan(v) || std::isnan(steer)) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
                "NaN trong lệnh điều khiển! v=%.2f s=%.2f", v, steer);
            return;
        }
        ackermann_msgs::msg::AckermannDriveStamped msg;
        msg.header.stamp         = now();
        msg.header.frame_id      = car_frame;
        msg.drive.speed          = v;
        msg.drive.steering_angle = steer;
        pub_drive->publish(msg);
    }

    // Vẽ quỹ đạo tốt nhất trên RViz (dùng pose snapshot để thread-safe)
    void publish_best_trajectory(double x, double y, double th) {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = map_frame;
        m.header.stamp    = now();
        m.ns   = "best_traj";
        m.id   = 0;
        m.type = visualization_msgs::msg::Marker::LINE_STRIP;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x   = 0.08;
        m.color.a   = 1.0;
        m.color.r   = 0.0;
        m.color.g   = 1.0;
        m.color.b   = 0.0;

        geometry_msgs::msg::Point p;
        p.x = x; p.y = y; p.z = 0.05;
        m.points.push_back(p);

        for (int t = 0; t < horizon; t++) {
            x  += nominal_control[t].v * std::cos(th) * dt;
            y  += nominal_control[t].v * std::sin(th) * dt;
            th += nominal_control[t].v * std::tan(nominal_control[t].steer) / WHEELBASE * dt;
            th  = normalize_angle(th);
            p.x = x; p.y = y;
            m.points.push_back(p);
        }
        pub_best_traj->publish(m);
    }

    // Vẽ toàn bộ waypoints trên RViz (chỉ gọi 1 lần lúc khởi động)
    void publish_waypoints_marker() {
        if (waypoints.empty()) return;
        visualization_msgs::msg::MarkerArray arr;
        visualization_msgs::msg::Marker m;
        m.header.frame_id = map_frame;
        m.header.stamp    = now();
        m.ns   = "waypoints";
        m.id   = 0;
        m.type = visualization_msgs::msg::Marker::POINTS;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = 0.05; m.scale.y = 0.05;
        m.color.a = 1.0; m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0;
        for (const auto& wp : waypoints) {
            geometry_msgs::msg::Point p;
            p.x = wp.x; p.y = wp.y; p.z = 0.0;
            m.points.push_back(p);
        }
        arr.markers.push_back(m);
        pub_waypoints->publish(arr);
    }

    // ============================================================
    // [6] TIỆN ÍCH
    // ============================================================

    // Chuẩn hóa góc về [-π, π]
    inline double normalize_angle(double a) {
        a = std::fmod(a + M_PI, 2.0 * M_PI);
        if (a < 0) a += 2.0 * M_PI;
        return a - M_PI;
    }
};

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MPPIController>();
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}
