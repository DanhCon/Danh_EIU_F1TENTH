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
 *   2. Tìm waypoint gần nhất & xây dựng local window
 *   3. Corridor Filter: phân loại LiDAR → tường / vật cản
 *   4. Curvature Profiling → tốc độ mục tiêu theo hình dạng đường
 *   5. Proactive Deceleration theo vật cản phía trước
 *   6. Kiểm tra độ trễ dữ liệu cảm biến
 *   7. MPPI sampling + tính cost song song (OpenMP)
 *   8. MPPI weight update → cập nhật nominal_control
 *   9. EMA smoothing → giảm chattering vô lăng
 *  10. CBF-QP Safety Shield → can thiệp tốc độ an toàn nếu có nguy cơ va chạm
 *  11. Publish lệnh điều khiển & ghi CSV log
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
#include <filesystem>
#include <iomanip>

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
struct ObsPt   { double x, y, r; };  // Điểm chướng ngại vật với bán kính nguy hiểm

// Kết quả phân loại của Corridor Filter
struct CorridorResult {
    std::vector<ObsPt> obs_pts;               // Danh sách chướng ngại vật (hệ map)
    double             min_front_dist = 999.0; // Khoảng cách chướng ngại vật gần nhất phía trước (m)
    int                obs_cnt        = 0;     // Số điểm LiDAR trong hành lang phía trước
    int                wall_cnt       = 0;     // Số điểm LiDAR phân loại là tường hai bên
};

// Chi tiết điểm số của mẫu MPPI tốt nhất (phục vụ log & chẩn đoán)
struct BestTrajectoryBreakdown {
    int    best_idx       = 0;
    double track_cost     = 0.0;
    double heading_cost   = 0.0;
    double speed_cost     = 0.0;
    double smooth_cost    = 0.0;
    double obs_cost       = 0.0;
    double terminal_cost  = 0.0;
    double collision_rate = 0.0;
};

// Kết quả đo đạc thời gian từng khâu của MPPI (Profiling)
struct ProfilingTimers {
    double t_prep_ms        = 0.0; // Chuẩn bị (snapshot, waypoint search, corridor, curvature)
    double t_sample_ms      = 0.0; // Sinh mẫu ngẫu nhiên Gaussian
    double t_cost_ms        = 0.0; // Đánh giá song song OpenMP 500 rollouts
    double t_update_ms      = 0.0; // Softmax reduction & cập nhật nominal_control
    double t_shield_ms      = 0.0; // EMA filtering + CBF-QP Safety Shield
    double t_total_ms       = 0.0; // Tổng thời gian tính toán thuần (ms, không sleep)
    double loop_interval_ms = 0.0; // Khoảng thời gian thực tế giữa 2 lần chạy control_loop (ms)
    double max_possible_hz  = 0.0; // Tần số tối đa lý thuyết CPU có thể gánh (1000 / t_total_ms)
    double actual_freq_hz   = 0.0; // Tần số thực tế đo được (1000 / loop_interval_ms)
};

// ============================================================
// MPPIController Node
// ============================================================
class MPPIController : public rclcpp::Node {
public:
    MPPIController() : Node("mppi_real_controller_node") {

        // ROS Parameters (có thể override từ launch file / command line)
        declare_parameter("horizon",            30);
        declare_parameter("num_samples",        500);
        declare_parameter("dt",                 0.05);
        declare_parameter("enable_console_log", false); // Mặc định tắt in console lúc chạy

        horizon            = get_parameter("horizon").as_int();
        num_samples        = get_parameter("num_samples").as_int();
        dt                 = get_parameter("dt").as_double();
        enable_console_log = get_parameter("enable_console_log").as_bool();

        // Pre-allocate MPPI buffers (tránh alloc trong vòng lặp)
        noise_buf.resize(num_samples, std::vector<Control>(horizon));
        costs_buf.resize(num_samples, 0.0);
        weights_buf.resize(num_samples, 0.0);
        nominal_control.resize(horizon, {0.0, 0.0});
        upd_v_buf.resize(horizon, 0.0);
        upd_s_buf.resize(horizon, 0.0);
        local_wps.reserve(WP_WINDOW);
        local_hdgs.reserve(WP_WINDOW);
        local_idxs.reserve(WP_WINDOW);

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
            "MPPI Real Controller started. WPs=%zu, H=%d, N=%d, dt=%.3fs (Console: %s)",
            waypoints.size(), horizon, num_samples, dt, (enable_console_log ? "BẬT" : "TẮT (êm ru)"));

        // Khởi tạo thư mục và file log CSV linh hoạt (hỗ trợ cả máy dev và xe thật fablab_01)
        try {
            const char* home_env = std::getenv("HOME");
            std::string log_dir = home_env ? (std::string(home_env) + "/mppi_logs") : "/home/danh/mppi_logs";
            std::filesystem::create_directories(log_dir);
            auto now_time = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now_time);
            std::stringstream ss;
            ss << log_dir << "/mppi_log_" 
               << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S") << ".csv";
            log_filepath = ss.str();
            csv_log.open(log_filepath);
            if (csv_log.is_open()) {
                // Header CSV bao gồm 25 cột cũ + các cột phân tích hiệu năng & tần số mới
                csv_log << "time,x,y,theta,v_cur,target_v,steer_cmd,v_cmd,exec_time_ms,is_stopped,is_stuck,front_blocked,"
                        << "odom_delay,lidar_delay,best_track_cost,best_obs_cost,collision_rate,cte,heading_err,curvature,"
                        << "steer_raw,min_cost,avg_cost,min_front_dist,obs_cnt,"
                        << "loop_interval_ms,max_freq_hz,actual_freq_hz,t_prep_ms,t_sample_ms,t_cost_ms,t_update_ms,t_shield_ms,"
                        << "cbf_active,cbf_v_limit,wall_cnt,w_sum,effective_samples\n";
                RCLCPP_INFO(get_logger(), "📁 Đang ghi telemetry log tại: %s", log_filepath.c_str());
            } else {
                RCLCPP_ERROR(get_logger(), "Không mở được file CSV log tại: %s", log_filepath.c_str());
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Lỗi khi tạo thư mục log: %s", e.what());
        }
    }

    ~MPPIController() {
        if (csv_log.is_open()) {
            csv_log.close();
            RCLCPP_INFO(get_logger(), "Đã đóng file CSV log: %s", log_filepath.c_str());
        }
        if (total_cycles > 0) {
            double avg_t = sum_comp_time_ms / total_cycles;
            double avg_max_hz = (avg_t > 0.001) ? (1000.0 / avg_t) : 0.0;
            double worst_max_hz = (max_comp_time_ms > 0.001) ? (1000.0 / max_comp_time_ms) : 0.0;

            std::stringstream ss;
            ss << "\n"
               << "====================================================================\n"
               << "📊 [MPPI PROFILING REPORT] TỔNG KẾT HIỆU NĂNG TÍNH TOÁN & TẦN SỐ\n"
               << "====================================================================\n"
               << "  * Tổng số chu kỳ chạy      : " << total_cycles << " chu kỳ\n"
               << "  * Thời gian tính toán thuần (Không bị kìm hãm bởi timer 20Hz):\n"
               << "      - Nhanh nhất (Min)      : " << std::fixed << std::setprecision(2) << min_comp_time_ms << " ms\n"
               << "      - Trung bình (Avg)      : " << avg_t << " ms  --> Tần số tối đa CPU gánh được: ~" << std::setprecision(1) << avg_max_hz << " Hz\n"
               << "      - Lâu nhất   (Max)      : " << std::setprecision(2) << max_comp_time_ms << " ms  --> Tần số tối đa lúc tải đỉnh : ~" << std::setprecision(1) << worst_max_hz << " Hz\n"
               << "  * Đánh giá khả năng tăng tần số điều khiển:\n"
               << "      - Mốc 30 Hz (budget 33.3 ms): " << (cycles_over_33ms == 0 ? "AN TOÀN" : "CÓ RỦI RO") << " (vượt budget: " << cycles_over_33ms << " lần)\n"
               << "      - Mốc 40 Hz (budget 25.0 ms): " << (cycles_over_25ms == 0 ? "AN TOÀN" : "CÓ RỦI RO") << " (vượt budget: " << cycles_over_25ms << " lần)\n"
               << "      - Mốc 50 Hz (budget 20.0 ms): " << (cycles_over_20ms == 0 ? "AN TOÀN" : "CÓ RỦI RO") << " (vượt budget: " << cycles_over_20ms << " lần)\n"
               << "  * KHUYẾN NGHỊ: " << (max_comp_time_ms < 18.0 ? "Có thể tự tin tăng lên 40 Hz - 50 Hz (dt = 0.02s - 0.025s)!" :
                                       (max_comp_time_ms < 28.0 ? "Có thể an toàn tăng lên 30 Hz (dt = 0.033s)!" : "Nên giữ ở mức 20 Hz.")) << "\n"
               << "  * File CSV chi tiết đã lưu  : " << log_filepath << "\n"
               << "====================================================================\n";
            std::cout << ss.str() << std::flush;
            try {
                RCLCPP_INFO(get_logger(), "%s", ss.str().c_str());
            } catch (...) {}
        }
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
    static constexpr double WHEELBASE     = 0.39;   // Chiều dài cơ sở (m)
    static constexpr double MAX_STEER_RAD = 0.4;  // Góc lái vật lý tối đa (rad)

    // Hằng số dọn dẹp Magic Numbers (Đã điều chỉnh cho waypoint cách đều 5cm = 0.05m)
    static constexpr int    WP_WINDOW                     = 200;   // Cửa sổ waypoint cục bộ (200 điểm * 0.05m = 10m)
    static constexpr int    WP_WINDOW_BACK                = 40;    // Số lượng waypoint lùi lại phía sau xe (40 điểm * 0.05m = 2.0m, tiến 160 điểm = 8.0m)
    static constexpr int    WP_SEARCH_BACK                = 30;    // Cửa sổ tìm kiếm waypoint lùi phía sau (30 điểm * 0.05m = 1.5m)
    static constexpr int    WP_SEARCH_FORWARD             = 60;    // Cửa sổ tìm kiếm waypoint tiến phía trước (60 điểm * 0.05m = 3.0m)
    static constexpr double TELEPORT_THRESHOLD_M          = 5.0;   // Ngưỡng phát hiện teleport (m)
    static constexpr double MIN_LIDAR_RANGE               = 0.1;   // Khoảng quét tối thiểu của LiDAR (m)
    static constexpr double MAX_LIDAR_RANGE               = 3.5;   // Khoảng quét tối đa của LiDAR (m)
    static constexpr double ODOM_TIMEOUT_S                = 0.5;   // Thời gian tối đa mất tín hiệu Odom (s)
    static constexpr double LIDAR_TIMEOUT_S               = 0.5;   // Thời gian tối đa mất tín hiệu LiDAR (s)
    static constexpr double DOWN_SAMPLE_ANGLE_DEG         = 1.0;   // Góc down-sample tia LiDAR (độ)
    static constexpr double CORRIDOR_MIN_DIST             = 0.1;   // Bỏ qua LiDAR quá gần xe (m)
    static constexpr double PROACTIVE_DECEL_MIN_DIST      = 0.5;   // Khoảng cách tối thiểu của giảm tốc chủ động (m)
    static constexpr double COLLISION_DISTANCE_M          = 0.2;   // Khoảng cách va chạm cực cận (m)
    static constexpr double TERMINAL_COST_MULTIPLIER      = 1.5;   // Hệ số nhân cho chi phí terminal
    static constexpr double STEER_DECAY_FACTOR            = 0.5;   // Hệ số suy giảm góc lái ở bước cuối
    static constexpr double VISUALIZATION_HEIGHT_Z        = 0.05;  // Chiều cao z để vẽ line trên RViz (m)

    // Tên frame ROS
    const std::string car_frame = "base_link";
    const std::string map_frame = "map";

    // ============================================================
    // [B] THAM SỐ ĐIỀU CHỈNH (TUNE)
    // ============================================================

    // -- MPPI Cost Weights --
    double lambda_    = 140.0;  // Softmax temperature: cao→đều, thấp→tham lam trajectory tốt nhất
    double w_track    = 20.0;   // Bám tâm đường (Đã tăng lên 20.0 theo tune 19/9)
    double w_heading  =  5.0;   // Song song đường đua (Đã giảm xuống 5.0 theo tune 19/9)
    double w_progress =  1.5;   // Khuyến khích tiến về phía trước (Chuẩn hóa cho bước điểm 5cm để giữ cân bằng với w_track)
    double w_obs      = 180.0;  // Né chướng ngại vật
    double w_smooth   = 15.5;   // Phạt bẻ lái/thay đổi tốc độ đột ngột (giảm rack-rack)
    double w_speed    =  8.0;   // Bám tốc độ mục tiêu

    // -- Tốc độ & Gia tốc --
    double target_speed_max = 3.0;  // Tốc độ tối đa (m/s)          [TUNE 19/9]
    double min_speed_curve  = 2.8;  // Tốc độ tối thiểu trong cua (m/s) [TUNE 19/9]
    double max_accel        = 2.5;  // Gia tốc tăng tốc tối đa (m/s²)
    double max_decel        = 2.61;  // Gia tốc phanh tối đa (m/s²)

    // -- Phát hiện cua (Curvature Profiling) --
    double curve_thresh        = 0.5;   // Độ cong ngưỡng cua gắt (1/m) [TUNE 19/9]
    int    speed_lookahead_wps = 100;  // Số waypoint nhìn trước để phát hiện cua (100 điểm * 0.05m = 5.0m)

    // -- Corridor Filter (phân loại LiDAR → tường / vật cản) --
    //
    //   Cách hoạt động: transform điểm LiDAR sang car frame,
    //   nếu nằm trong hành lang phía trước → vật cản nguy hiểm,
    //   còn lại (2 bên, phía sau) → tường bình thường.
    //
    //   Ưu điểm so với Wall Filter cũ: không cần x0/y0 chính xác
    //   từ Particle Filter, chỉ cần heading theta0.
    //
    double corridor_half_w = 0.45;  // Nửa chiều rộng hành lang (m) - Giảm xuống 0.45m để tránh nuốt cả tường 2 bên vào vùng vật cản
    double corridor_max_d  = 6.0;   // Chiều dài hành lang nhìn trước (m)
    double r_obstacle      = 0.40;  // Bán kính nguy hiểm của vật cản (m) - Tăng lên 0.45m để ép MPPI né xa và sớm hơn nữa
    double r_wall          = 0.30;  // Bán kính nguy hiểm của tường 2 bên (m) - Giảm xuống 0.20m để xe dám chạy sát tường khi lách vật cản
    double collision_cost  = 200.0; // Phạt cực nặng nếu quẹt trúng vật cản

    // -- [TEST MODE] Bật/tắt né tránh vật cản --
    //   false = TẮT toàn bộ phần NÉ (corridor cost, proactive decel).
    //   VẪN GIỮ NGUYÊN: front_blocked (phanh khẩn cấp), anti-stuck watchdog,
    //   escape maneuver, dừng an toàn khi mất tín hiệu.
    bool enable_obstacle_avoidance = false;

    // -- Proactive Deceleration (giảm tốc sớm khi có vật cản) --
    double obs_decel_start_dist = 1.5;   // Giảm xuống 1.5m để tránh giảm tốc quá sớm làm ngắn tầm nhìn MPPI
    double obs_decel_min_factor = 0.60;  // Giới hạn giảm tốc ở mức 60% tốc độ để giữ tầm nhìn xa và ổn định động cơ

    // -- CBF-QP Safety Filter Parameters (Thay thế cơ chế phanh khẩn cấp & lùi cũ, tham khảo code_chay_cbf_thuong) --
    bool   enable_cbf          = true;  // Bật bộ lọc an toàn CBF
    double cbf_d_min           = 0.35;  // Khoảng cách an toàn tối thiểu dừng xe (m)
    double cbf_gamma           = 2.5;   // Hệ số rào chắn CBF (độ nhạy giảm tốc khi tiến gần)
    double cbf_fov_cutoff_deg  = 15.0;  // Góc nón quan sát an toàn phía trước (+/- độ)

    // -- EMA Output Smoothing (giảm nhiễu MPPI → bớt rack-rack) --
    double alpha_v = 0.15;  // Hệ số EMA tốc độ:    nhỏ → mượt hơn, trễ hơn
    double alpha_s = 0.70;  // Hệ số EMA góc lái:   nhỏ → mượt hơn, trễ hơn 

    // -- MPPI Exploration Noise --
    double sigma_v = 1.5;   // Độ lệch chuẩn nhiễu tốc độ MPPI 
    double sigma_s = 0.20;  // Độ lệch chuẩn góc lái - Đặt ở 0.20 để lách êm hơn, tránh trượt lốp

    // -- MPPI Horizon (set từ ROS param) --
    int    horizon, num_samples;
    double dt;
    bool   enable_console_log = false;

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
    std::vector<double>               upd_v_buf;       // Buffer cộng dồn vận tốc (tránh alloc)
    std::vector<double>               upd_s_buf;       // Buffer cộng dồn góc lái (tránh alloc)

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

    std::mt19937 rng;
    std::ofstream csv_log;
    std::string   log_filepath;

    // Thống kê hiệu năng toàn bộ quá trình chạy (đo thuần túy không ràng buộc bởi timer 20Hz)
    size_t total_cycles           = 0;
    double sum_comp_time_ms       = 0.0;
    double min_comp_time_ms       = 9999.0;
    double max_comp_time_ms       = 0.0;
    size_t cycles_over_20ms       = 0; // Vượt 20ms (mốc 50 Hz)
    size_t cycles_over_25ms       = 0; // Vượt 25ms (mốc 40 Hz)
    size_t cycles_over_33ms       = 0; // Vượt 33.3ms (mốc 30 Hz)
    size_t cycles_over_50ms       = 0; // Vượt 50ms (mốc 20 Hz)
    std::chrono::high_resolution_clock::time_point last_loop_tp;
    bool has_last_loop_tp         = false;

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
        std::string active_path = path;
        if (!file.is_open()) {
            std::vector<std::string> fallbacks = {
                "src/f1tenth_waypoint.csv",
                "src/f1tenth_waypoint_smooth.csv"
            };
            for (const auto& fb : fallbacks) {
                file.open(fb);
                if (file.is_open()) {
                    active_path = fb;
                    RCLCPP_WARN(get_logger(), "Không tìm thấy '%s', tự động chuyển sang file waypoint: %s", path.c_str(), fb.c_str());
                    break;
                }
            }
        }
        if (!file.is_open()) {
            RCLCPP_FATAL(get_logger(), "Không mở được file waypoint: %s (cũng không tìm thấy file fallback)", path.c_str());
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

        // Tính heading và curvature bằng span phù hợp với bước điểm 5cm (span 10 điểm * 0.05m = 0.5m mỗi bên, tổng 1.0m)
        int w = static_cast<int>(waypoints.size());
        waypoint_headings.resize(w);
        waypoint_curvatures.resize(w);
        const int CURV_SPAN = (w >= 40) ? 10 : std::max(1, w / 4);
        for (int i = 0; i < w; i++) {
            auto& p1 = waypoints[(i - CURV_SPAN + w) % w];
            auto& p2 = waypoints[i];
            auto& p3 = waypoints[(i + CURV_SPAN) % w];
            waypoint_headings[i] = std::atan2(p3.y - p1.y, p3.x - p1.x);
            double dx1 = p2.x - p1.x, dy1 = p2.y - p1.y;
            double dx2 = p3.x - p2.x, dy2 = p3.y - p2.y;
            double l1 = std::hypot(dx1, dy1), l2 = std::hypot(dx2, dy2);
            double l3 = std::hypot(p3.x - p1.x, p3.y - p1.y);
            waypoint_curvatures[i] = (l1 * l2 * l3 > 1e-9)
                ? 4.0 * (dx1 * dy2 - dy1 * dx2) / (l1 * l2 * l3)
                : 0.0;
        }
        RCLCPP_INFO(get_logger(), "Đã tải %d waypoints (Curvature span = %d điểm).", w, CURV_SPAN);
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
        // Lấy transform từ laser frame → car_frame ("base_link") để giữ tọa độ vật cản ở local frame
        geometry_msgs::msg::TransformStamped tf;
        try {
            tf = tf_buffer->lookupTransform(car_frame, msg->header.frame_id, tf2::TimePointZero);
        } catch (...) { return; }

        auto& q = tf.transform.rotation;
        double yaw = std::atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z));
        double tx  = tf.transform.translation.x;
        double ty  = tf.transform.translation.y;

        // Down-sample theo góc DOWN_SAMPLE_ANGLE_DEG để giảm khối lượng tính toán
        int step = std::max(1, static_cast<int>((DOWN_SAMPLE_ANGLE_DEG * M_PI / 180.0) / msg->angle_increment));

        std::vector<Point2D> temp;
        temp.reserve(msg->ranges.size() / step + 1);
        for (size_t i = 0; i < msg->ranges.size(); i += step) {
            double r = msg->ranges[i];
            if (!std::isnormal(r) || r < MIN_LIDAR_RANGE || r > MAX_LIDAR_RANGE) continue;
            double angle = msg->angle_min + i * msg->angle_increment;
            double px = r * std::cos(angle);
            double py = r * std::sin(angle);
            temp.push_back({
                tx + px * std::cos(yaw) - py * std::sin(yaw),
                ty + px * std::sin(yaw) + py * std::cos(yaw)
            });
        }

        // Swap O(1) thay vì copy O(n) - Lúc này map_obstacles chứa tọa độ local trong hệ base_link
        std::lock_guard<std::mutex> lock(obs_mutex);
        std::swap(map_obstacles, temp);
        obstacle_stamp = now();
    }

    // =========================================================================
    // [PHẦN 4.1] TÌM WAYPOINT GẦN XE NHẤT (NEAREST WAYPOINT SEARCH)
    // =========================================================================
    /**
     * 🎯 CHỨC NĂNG:
     *    Xác định waypoint trên raceline có khoảng cách Euclid gần xe nhất.
     * 📥 INPUT:
     *    - x, y: Tọa độ hiện tại của xe trong hệ quy chiếu map (m).
     * 📤 OUTPUT:
     *    - int: Chỉ số (index) của waypoint gần nhất trong mảng toàn cục waypoints.
     * ⚙️ QUY TRÌNH XỬ LÝ:
     *    1. Quét tìm trong cửa sổ hẹp [-20, +40] waypoints quanh vị trí vòng trước (O(60)) để tối ưu CPU.
     *    2. Nếu khoảng cách > TELEPORT_THRESHOLD_M (5.0m), kích hoạt tìm kiếm toàn bộ raceline (O(N)).
     *    3. Cập nhật và lưu lại cache last_nearest_wp cho vòng sau.
     */
    int find_nearest_waypoint(double x, double y) {
        int base_wp = last_nearest_wp;
        int best_wp = base_wp;
        double min_d = 9999.0;
        int w = static_cast<int>(waypoints.size());
        for (int di = -WP_SEARCH_BACK; di <= WP_SEARCH_FORWARD; di++) {
            int idx = ((base_wp + di) % w + w) % w;
            double d = std::hypot(waypoints[idx].x - x, waypoints[idx].y - y);
            if (d < min_d) { min_d = d; best_wp = idx; }
        }
        if (min_d > TELEPORT_THRESHOLD_M) { // Teleport recovery
            for (int i = 0; i < w; i++) {
                double d = std::hypot(waypoints[i].x - x, waypoints[i].y - y);
                if (d < min_d) { min_d = d; best_wp = i; }
            }
        }
        last_nearest_wp = best_wp;
        return best_wp;
    }

    // =========================================================================
    // [PHẦN 4.2] XÂY DỰNG CỬA SỔ WAYPOINT CỤC BỘ (LOCAL WINDOW)
    // =========================================================================
    /**
     * 🎯 CHỨC NĂNG:
     *    Trích xuất một cửa sổ con 80 waypoints xung quanh xe (-15 lùi, +65 tiến)
     *    để MPPI tính toán sai số bám đường siêu tốc mà không cần duyệt qua toàn bộ raceline.
     * 📥 INPUT:
     *    - nearest_wp: Index waypoint gần xe nhất vừa tìm được.
     * 📤 OUTPUT:
     *    - int: Vị trí index của waypoint gần xe nhất TRONG CỬA SỔ CỤC BỘ (local_nearest).
     *           Trả về -1 nếu có lỗi. Cập nhật các buffer local_wps, local_hdgs, local_idxs.
     * ⚙️ QUY TRÌNH XỬ LÝ:
     *    1. Xóa sạch các vector cục bộ local_wps, local_hdgs, local_idxs.
     *    2. Quét từ [-WP_WINDOW_BACK, WP_WINDOW - WP_WINDOW_BACK) và lấy tọa độ, heading, index gốc.
     *    3. Xác định vị trí local_nearest đại diện cho xe trong cửa sổ này.
     */
    int build_local_waypoint_window(int nearest_wp) {
        local_wps.clear(); local_hdgs.clear(); local_idxs.clear();
        for (int i = -WP_WINDOW_BACK; i < WP_WINDOW - WP_WINDOW_BACK; i++) {
            int idx = ((nearest_wp + i) % (int)waypoints.size() + (int)waypoints.size()) % (int)waypoints.size();
            local_wps.push_back(waypoints[idx]);
            local_hdgs.push_back(waypoint_headings[idx]);
            local_idxs.push_back(idx);
        }
        int local_nearest = WP_WINDOW_BACK;
        return local_nearest;
    }

    // =========================================================================
    // [PHẦN 4.3] BỘ LỌC HÀNH LANG LIDAR (CORRIDOR FILTER)
    // =========================================================================
    /**
     * 🎯 CHỨC NĂNG:
     *    Phân loại các điểm quét LiDAR thành hai nhóm:
     *    - Vật cản nguy hiểm phía trước (trong hành lang xe chạy): Gán bán kính né lớn r_obstacle.
     *    - Tường hai bên đường đua: Gán bán kính né nhỏ r_wall để xe dám chạy sát tường.
     * 📥 INPUT:
     *    - x, y, th: Pose hiện tại của xe trong hệ tọa độ map.
     *    - raw_obs: Mảng các điểm LiDAR đã đưa về hệ trục gắn với xe (base_link).
     * 📤 OUTPUT:
     *    - CorridorResult: Chứa danh sách obs_pts (hệ map), cự ly vật cản min_front_dist, và số lượng điểm.
     * ⚙️ QUY TRÌNH XỬ LÝ:
     *    1. Nếu bật né tránh (enable_obstacle_avoidance = true):
     *       - Điểm trong hành lang (|y| < corridor_half_w, x > CORRIDOR_MIN_DIST): Là VẬT CẢN.
     *       - Điểm ngoài hành lang: Là TƯỜNG.
     *       - Chiếu điểm sang hệ tọa độ map bằng phép xoay cos(th), sin(th) để phục vụ MPPI rollout.
     *    2. Nếu tắt né tránh (enable_obstacle_avoidance = false):
     *       - Tính nhanh cự ly gần nhất min_front_dist và obs_cnt trực tiếp trên hệ trục xe (tiết kiệm CPU).
     */
    CorridorResult filter_lidar_corridor(double x, double y, double th, const std::vector<Point2D>& raw_obs) {
        CorridorResult res;
        if (enable_obstacle_avoidance) {
            res.obs_pts.reserve(raw_obs.size());
            for (const auto& pt : raw_obs) {
                double dx_l = pt.x; // trục X xe (phía trước)
                double dy_l = pt.y; // trục Y xe (bên trái)
                double r;
                if (dx_l > CORRIDOR_MIN_DIST && dx_l < corridor_max_d && std::abs(dy_l) < corridor_half_w) {
                    r = r_obstacle;
                    res.obs_cnt++;
                    if (dx_l < res.min_front_dist) res.min_front_dist = dx_l;
                } else {
                    r = r_wall;
                    res.wall_cnt++;
                }
                double mx = x + pt.x * std::cos(th) - pt.y * std::sin(th);
                double my = y + pt.x * std::sin(th) + pt.y * std::cos(th);
                res.obs_pts.push_back({mx, my, r});
            }
            if (enable_console_log) {
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
                    "Corridor: wall=%d(r=%.2f) | obs=%d(r=%.2f) | front=%.2fm",
                    res.wall_cnt, r_wall, res.obs_cnt, r_obstacle, res.min_front_dist);
            }
        } else {
            for (const auto& pt : raw_obs) {
                if (pt.x > CORRIDOR_MIN_DIST && pt.x < corridor_max_d && std::abs(pt.y) < corridor_half_w) {
                    res.obs_cnt++;
                    if (pt.x < res.min_front_dist) res.min_front_dist = pt.x;
                } else {
                    res.wall_cnt++;
                }
            }
        }
        return res;
    }

    // =========================================================================
    // [PHẦN 4.4] TÍNH TỐC ĐỘ MỤC TIÊU THEO ĐỘ CONG & VẬT CẢN (CURVATURE PROFILING)
    // =========================================================================
    /**
     * 🎯 CHỨC NĂNG:
     *    Tự động tính toán tốc độ chạy tối ưu theo hình học khúc cua phía trước
     *    và chủ động giảm tốc sớm nếu phát hiện vật cản tiến gần.
     * 📥 INPUT:
     *    - nearest_wp: Index waypoint gần xe nhất.
     *    - min_front_dist: Khoảng cách tới vật cản gần nhất phía trước (từ Corridor Filter).
     * 📤 OUTPUT:
     *    - max_c_out: Độ cong lớn nhất phía trước (1/m).
     *    - double: Vận tốc mục tiêu target_v (m/s) đã qua giới hạn gia tốc an toàn.
     * ⚙️ QUY TRÌNH XỬ LÝ:
     *    1. Quét nhìn trước speed_lookahead_wps (25 waypoints) để tìm độ cong lớn nhất max_c.
     *    2. Hạ tốc độ từ target_speed_max (3.0 m/s) xuống min_speed_curve (2.8 m/s) nếu max_c > curve_thresh.
     *    3. Nếu bật né tránh và có vật cản gần hơn obs_decel_start_dist (1.5m), giảm tốc chủ động.
     *    4. Giới hạn gia tốc tăng/giảm [last - max_decel*dt, last + max_accel*dt] để động cơ ổn định.
     */
    double compute_target_velocity(int nearest_wp, double min_front_dist, double& max_c_out) {
        double max_c = 0.0;
        for (int i = 0; i < speed_lookahead_wps; i++) {
            double c = std::abs(waypoint_curvatures[(nearest_wp + i) % (int)waypoints.size()]);
            if (c > max_c) max_c = c;
        }
        max_c_out = max_c;

        double speed_factor = (max_c > curve_thresh)
            ? std::max(0.0, 1.0 - (max_c - curve_thresh) / curve_thresh)
            : 1.0;
        double target_v = min_speed_curve + (target_speed_max - min_speed_curve) * speed_factor;

        if (enable_obstacle_avoidance && min_front_dist < obs_decel_start_dist) {
            double f = obs_decel_min_factor
                + (1.0 - obs_decel_min_factor) * ((min_front_dist - PROACTIVE_DECEL_MIN_DIST) / (obs_decel_start_dist - PROACTIVE_DECEL_MIN_DIST));
            f = std::max(obs_decel_min_factor, std::min(1.0, f));
            target_v = std::min(target_v, target_speed_max * f);
            if (enable_console_log) {
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 200,
                    "Proactive Decel: dist=%.2fm → tgt_v=%.2fm/s", min_front_dist, target_v);
            }
        }

        target_v = std::min(last_target_speed + max_accel * dt, target_v);
        target_v = std::max(last_target_speed - max_decel * dt, target_v);
        last_target_speed = target_v;
        return target_v;
    }

    // =========================================================================
    // [PHẦN 4.5] SINH NHIỄU KHÁM PHÁ MPPI (SAMPLING NOISE)
    // =========================================================================
    /**
     * 🎯 CHỨC NĂNG:
     *    Tạo ma trận nhiễu phân phối chuẩn Gaussian ngẫu nhiên cho N=500 quỹ đạo, H=30 bước.
     * 📥 INPUT:
     *    - Sử dụng rng, sigma_v, sigma_s, horizon, num_samples nội tại.
     * 📤 OUTPUT:
     *    - Cập nhật trực tiếp vào buffer noise_buf[num_samples][horizon].
     */
    void sample_mppi_noise() {
        std::normal_distribution<double> dist_v(0.0, sigma_v);
        std::normal_distribution<double> dist_s(0.0, sigma_s);
        for (int n = 0; n < num_samples; n++) {
            for (int t = 0; t < horizon; t++) {
                noise_buf[n][t].v     = dist_v(rng);
                noise_buf[n][t].steer = dist_s(rng);
            }
        }
    }

    // =========================================================================
    // [PHẦN 4.6] GIẢ LẬP SONG SONG & TÍNH CHI PHÍ MPPI (PARALLEL ROLLOUTS - OPENMP)
    // =========================================================================
    /**
     * 🎯 CHỨC NĂNG:
     *    Mô phỏng 500 quỹ đạo xe ảo chạy đồng thời trên CPU qua OpenMP bằng mô hình xe đạp
     *    và chấm điểm chi phí tổng hợp cho từng quỹ đạo.
     * 📥 INPUT:
     *    - x, y, th: Pose khởi đầu của xe (hệ map).
     *    - target_v: Vận tốc mục tiêu của chu kỳ.
     *    - obs_pts: Danh sách vật cản (hệ map).
     *    - nearest_wp: Index waypoint gần xe nhất.
     *    - local_nearest: Index của xe trong cửa sổ waypoint cục bộ.
     * 📤 OUTPUT:
     *    - int: Số lượng quỹ đạo bị va chạm (collision_samples).
     *    - Cập nhật trực tiếp vào mảng costs_buf[num_samples].
     * ⚙️ QUY TRÌNH XỬ LÝ:
     *    1. Phân luồng OpenMP song song động cho 500 mẫu thử.
     *    2. Mỗi luồng tích phân mô hình xe đạp qua 30 bước thời gian dt = 0.05s.
     *    3. Tính tổng các thành phần chi phí: Sai số tim đường (trk), sai số hướng (hdg),
     *       sai số tốc độ (spd), độ mượt (smo), vật cản (obs) và terminal progress.
     */
    int evaluate_rollouts_parallel(double x, double y, double th, double target_v,
                                   const std::vector<ObsPt>& obs_pts,
                                   int nearest_wp, int local_nearest) {
        double v_min = 0.0;
        double v_max = target_v;
        int collision_samples = 0;

        #pragma omp parallel for schedule(dynamic) reduction(+:collision_samples)
        for (int n = 0; n < num_samples; n++) {
            double px = x, py = y, pth = th;
            double trk = 0.0, hdg = 0.0, spd = 0.0, smo = 0.0, obs = 0.0;
            int    g_idx = nearest_wp;
            double pv_prv = nominal_control[0].v;
            double ps_prv = nominal_control[0].steer;
            double terminal = 0.0;
            bool collided = false;

            for (int t = 0; t < horizon; t++) {
                double pv = std::max(v_min, std::min(v_max, nominal_control[t].v + noise_buf[n][t].v));
                double ps = std::max(-MAX_STEER_RAD, std::min(MAX_STEER_RAD, nominal_control[t].steer + noise_buf[n][t].steer));

                // Mô hình xe đạp động học (Bicycle model)
                px  += pv * std::cos(pth) * dt;
                py  += pv * std::sin(pth) * dt;
                pth += pv * std::tan(ps) / WHEELBASE * dt;
                pth  = normalize_angle(pth);

                // Sai số tim đường & góc hướng so với waypoint gần nhất
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

                // Chi phí bám tốc độ & độ mượt
                spd += (pv - target_v) * (pv - target_v);
                if (t > 0) smo += (ps - ps_prv) * (ps - ps_prv) + (pv - pv_prv) * (pv - pv_prv);
                pv_prv = pv; ps_prv = ps;

                // Chi phí vật cản
                if (enable_obstacle_avoidance) {
                    double max_pen = 0.0, min_abs_d = 999.0;
                    for (const auto& o : obs_pts) {
                        double odx = px - o.x; if (std::abs(odx) > o.r) continue;
                        double ody = py - o.y; if (std::abs(ody) > o.r) continue;
                        double d   = std::hypot(odx, ody);
                        if (d < min_abs_d) min_abs_d = d;
                        if (d < o.r) { double p = (o.r - d) * (o.r - d); if (p > max_pen) max_pen = p; }
                    }
                    obs += max_pen;
                    if (min_abs_d < COLLISION_DISTANCE_M) {
                        obs += collision_cost;
                        collided = true;
                    }
                }

                // Chi phí terminal bước cuối
                if (t == horizon - 1) {
                    terminal += TERMINAL_COST_MULTIPLIER * (w_track * min_d2 + w_heading * herr * herr);
                    int prog = (g_idx - local_idxs[local_nearest] + (int)waypoints.size()) % (int)waypoints.size();
                    double prog_v = (prog > (int)waypoints.size() / 2) ? (double)(waypoints.size() - prog) : -(double)prog;
                    terminal += w_progress * prog_v;
                }
            }
            costs_buf[n] = terminal + w_track * trk + w_heading * hdg + w_speed * spd + w_smooth * smo + w_obs * obs;
            if (collided) collision_samples++;
        }
        return collision_samples;
    }

    // =========================================================================
    // [PHẦN 4.7] CẬP NHẬT ĐIỀU KHIỂN TỐI ƯU MPPI (WEIGHT UPDATE)
    // =========================================================================
    /**
     * 🎯 CHỨC NĂNG:
     *    Tính toán trọng số xác suất Softmax theo nguyên lý Feynman-Kac / MPPI
     *    và cập nhật lại quỹ đạo điều khiển danh nghĩa nominal_control.
     * 📥 INPUT:
     *    - min_cost: Chi phí nhỏ nhất trong các mẫu.
     *    - target_v: Vận tốc trần mục tiêu.
     * 📤 OUTPUT:
     *    - double: Tổng trọng số w_sum (dùng phát hiện weight collapse). Cập nhật nominal_control.
     * ⚙️ QUY TRÌNH XỬ LÝ:
     *    1. Tính w_n = exp(-(cost_n - min_cost) / lambda).
     *    2. Nếu w_sum > 1e-10: Cập nhật delta u = (sum w_n * noise_n) / w_sum.
     *    3. Giới hạn clamp vào dải vật lý: [0, target_v] và [-MAX_STEER_RAD, MAX_STEER_RAD].
     */

    double update_nominal_control(double min_cost, double target_v, double& effective_samples_out) {
        double v_min = 0.0;
        double v_max = target_v;
        double w_sum = 0.0;
        for (int n = 0; n < num_samples; n++) {
            weights_buf[n] = std::exp(-(costs_buf[n] - min_cost) / lambda_);
            w_sum += weights_buf[n];
        }

        double sum_w_norm_sq = 0.0;
        if (w_sum > 1e-10) {
            std::fill(upd_v_buf.begin(), upd_v_buf.end(), 0.0);
            std::fill(upd_s_buf.begin(), upd_s_buf.end(), 0.0);
            for (int n = 0; n < num_samples; n++) {
                double w = weights_buf[n];
                double w_norm = w / w_sum;
                sum_w_norm_sq += w_norm * w_norm;
                for (int t = 0; t < horizon; t++) {
                    upd_v_buf[t] += w * noise_buf[n][t].v;
                    upd_s_buf[t] += w * noise_buf[n][t].steer;
                }
            }
            effective_samples_out = (sum_w_norm_sq > 1e-12) ? (1.0 / sum_w_norm_sq) : 0.0;
            for (int t = 0; t < horizon; t++) {
                nominal_control[t].v     = std::max(v_min, std::min(v_max, nominal_control[t].v + upd_v_buf[t] / w_sum));
                nominal_control[t].steer = std::max(-MAX_STEER_RAD, std::min(MAX_STEER_RAD, nominal_control[t].steer + upd_s_buf[t] / w_sum));
            }
        } else {
            effective_samples_out = 0.0;
            if (enable_console_log) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                    "MPPI weight collapse (w_sum≈0)! Giữ nguyên nominal control.");
            }
            for (int t = 0; t < horizon; t++) {
                nominal_control[t].v     = std::max(v_min, std::min(v_max, nominal_control[t].v));
                nominal_control[t].steer = std::max(-MAX_STEER_RAD, std::min(MAX_STEER_RAD, nominal_control[t].steer));
            }
        }
        return w_sum;
    }

    // =========================================================================
    // [PHẦN 4.8] BỘ LỌC LÀM MƯỢT TÍN HIỆU ĐIỀU KHIỂN (EMA SMOOTHING)
    // =========================================================================
    /**
     * 🎯 CHỨC NĂNG:
     *    Làm mượt tín hiệu vận tốc và góc lái bằng bộ lọc Exponential Moving Average
     *    để triệt tiêu hiện tượng rung lắc vô lăng (chattering/rack-rack).
     * 📥 INPUT:
     *    - raw_v, raw_steer: Lệnh thô tại bước đầu tiên của nominal_control.
     * 📤 OUTPUT:
     *    - Control: Lệnh điều khiển đã làm mượt {last_ema_v, last_ema_steer}.
     */
    Control apply_ema_filter(double raw_v, double raw_steer) {
        last_ema_v     = alpha_v * raw_v     + (1.0 - alpha_v) * last_ema_v;
        last_ema_steer = alpha_s * raw_steer + (1.0 - alpha_s) * last_ema_steer;
        return {last_ema_v, last_ema_steer};
    }

    // =========================================================================
    // [PHẦN 4.9] LÁ CHẮN AN TOÀN RÀO CHẮN ĐIỀU KHIỂN (CBF SAFETY SHIELD)
    // =========================================================================
    /**
     * 🎯 CHỨC NĂNG:
     *    Lá chắn bảo vệ khẩn cấp dựa trên Control Barrier Function (CBF-QP)
     *    can thiệp hạ tốc độ hoặc dừng tuyệt đối nếu có nguy cơ đâm va trực diện.
     * 📥 INPUT:
     *    - raw_obs: Đám mây điểm LiDAR trong hệ base_link của xe.
     *    - obs_fresh: Trạng thái tươi mới của cảm biến (không bị delay quá 0.5s).
     *    - v_cmd, steer_cmd: Lệnh điều khiển mong muốn sau bộ lọc EMA.
     *    - target_v: Giới hạn tốc độ mục tiêu.
     * 📤 OUTPUT:
     *    - Control: Lệnh an toàn {final_v, final_steer}.
     *    - cbf_active_out: true nếu CBF can thiệp hạ tốc độ.
     *    - cbf_v_limit_out: tốc độ trần do CBF tính toán.
     * ⚙️ QUY TRÌNH XỬ LÝ:
     *    1. Quét các điểm LiDAR phía trước xe trong nón quan sát (+/- cbf_fov_cutoff_deg = 15 độ).
     *    2. Tính toán giới hạn vận tốc an toàn theo định lý Nagumo: limit_i = (gamma * (r_i - d_min)) / cos(phi_i).
     *    3. Giới hạn v_safe = max(0, min(v_cmd, min_limit)).
     *    4. Nếu cự ly r_i <= d_min (0.35m), limit <= 0 -> v_safe = 0.0 (Dừng hoàn toàn tại chỗ).
     */
    Control apply_cbf_safety_shield(const std::vector<Point2D>& raw_obs, bool obs_fresh,
                                   double v_cmd, double steer_cmd, double target_v,
                                   bool& cbf_active_out, double& cbf_v_limit_out) {
        double final_v     = v_cmd;
        double final_steer = steer_cmd;
        cbf_active_out  = false;
        cbf_v_limit_out = target_v;

        if (enable_cbf && obs_fresh) {
            double v_cbf_max = target_v;
            double min_cbf_dist = 999.0;
            int cbf_danger_count = 0;

            for (const auto& pt : raw_obs) {
                if (pt.x <= 0.05) continue; // Bỏ qua điểm thân xe
                double angle_deg = std::abs(std::atan2(pt.y, pt.x) * 180.0 / M_PI);
                if (angle_deg <= cbf_fov_cutoff_deg) {
                    double r_i = std::hypot(pt.x, pt.y);
                    if (r_i < min_cbf_dist) min_cbf_dist = r_i;
                    double cos_phi = pt.x / r_i;

                    double limit_i = (cbf_gamma * (r_i - cbf_d_min)) / cos_phi;
                    if (limit_i < v_cbf_max) {
                        v_cbf_max = limit_i;
                        cbf_danger_count++;
                    }
                }
            }
            cbf_v_limit_out = v_cbf_max;

            double v_safe = std::max(0.0, std::min(final_v, v_cbf_max));
            if (v_safe < final_v - 0.05) {
                cbf_active_out = true;
                if (enable_console_log) {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500,
                        "🛡️ [CBF SHIELD] Giảm tốc an toàn: v_cmd=%.2f -> v_safe=%.2f m/s (min_dist=%.2fm, d_min=%.2fm, gamma=%.1f, rays=%d)",
                        final_v, v_safe, min_cbf_dist, cbf_d_min, cbf_gamma, cbf_danger_count);
                }
                last_target_speed = std::min(last_target_speed, v_safe);
                if (v_safe <= 0.01) {
                    last_ema_v = 0.0; // Triệt tiêu độ trễ của EMA khi dừng hẳn
                    last_target_speed = 0.0;
                }
            }
            final_v = v_safe;
        }
        return {final_v, final_steer};
    }

    // =========================================================================
    // [PHẦN 4.10] TỊNH TIẾN CỬA SỔ THỜI GIAN (SHIFT HORIZON)
    // =========================================================================
    /**
     * 🎯 CHỨC NĂNG:
     *    Dịch chuyển mảng điều khiển danh nghĩa tiến lên 1 bước để làm điểm khởi động (warm-start)
     *    cho vòng lặp 20Hz kế tiếp.
     * 📥 INPUT:
     *    - target_v: Vận tốc gán cho bước cuối cùng.
     * 📤 OUTPUT:
     *    - Cập nhật trực tiếp vào nominal_control.
     */
    void shift_horizon(double target_v) {
        for (int t = 0; t < horizon - 1; t++) nominal_control[t] = nominal_control[t + 1];
        nominal_control[horizon - 1].v     = target_v;
        nominal_control[horizon - 1].steer = nominal_control[horizon - 2].steer * STEER_DECAY_FACTOR;
    }

    // =========================================================================
    // [PHẦN 4.11] TÍNH CHI TIẾT ĐIỂM SỐ QUỸ ĐẠO TỐT NHẤT (BEST TRAJECTORY BREAKDOWN)
    // =========================================================================
    /**
     * 🎯 CHỨC NĂNG:
     *    Mô phỏng lại quỹ đạo tốt nhất để bóc tách từng thành phần điểm phạt
     *    (bám đường, hướng, tốc độ, mượt, né vật cản, terminal) phục vụ phân tích.
     */
    BestTrajectoryBreakdown compute_best_trajectory_breakdown(
            int best_idx, int local_nearest, const std::vector<ObsPt>& obs_pts,
            double target_v, double x, double y, double th, int collision_samples) {
        BestTrajectoryBreakdown b;
        b.best_idx = best_idx;
        b.collision_rate = (double)collision_samples / num_samples * 100.0;
        double v_min = 0.0;
        double v_max = target_v;

        double px = x, py = y, pth = th;
        double pv_prv = nominal_control[0].v;
        double ps_prv = nominal_control[0].steer;

        for (int t = 0; t < horizon; t++) {
            double pv = std::max(v_min, std::min(v_max, nominal_control[t].v + noise_buf[best_idx][t].v));
            double ps = std::max(-MAX_STEER_RAD, std::min(MAX_STEER_RAD, nominal_control[t].steer + noise_buf[best_idx][t].steer));
            px  += pv * std::cos(pth) * dt;
            py  += pv * std::sin(pth) * dt;
            pth += pv * std::tan(ps) / WHEELBASE * dt;
            pth  = normalize_angle(pth);

            double min_d2 = 999.0; int min_wi = 0;
            for (int wi = 0; wi < (int)local_wps.size(); wi++) {
                double d2 = (px - local_wps[wi].x) * (px - local_wps[wi].x) + (py - local_wps[wi].y) * (py - local_wps[wi].y);
                if (d2 < min_d2) { min_d2 = d2; min_wi = wi; }
            }
            b.track_cost += min_d2;
            double herr = normalize_angle(pth - local_hdgs[min_wi]);
            b.heading_cost += herr * herr;
            b.speed_cost += (pv - target_v) * (pv - target_v);
            if (t > 0) b.smooth_cost += (ps - ps_prv) * (ps - ps_prv) + (pv - pv_prv) * (pv - pv_prv);
            pv_prv = pv; ps_prv = ps;

            if (enable_obstacle_avoidance) {
                double max_pen = 0.0, min_abs_d = 999.0;
                for (const auto& o : obs_pts) {
                    double odx = px - o.x; if (std::abs(odx) > o.r) continue;
                    double ody = py - o.y; if (std::abs(ody) > o.r) continue;
                    double d   = std::hypot(odx, ody);
                    if (d < min_abs_d) min_abs_d = d;
                    if (d < o.r) { double p = (o.r - d) * (o.r - d); if (p > max_pen) max_pen = p; }
                }
                b.obs_cost += max_pen;
                if (min_abs_d < COLLISION_DISTANCE_M) b.obs_cost += collision_cost;
            }
            if (t == horizon - 1) {
                b.terminal_cost += TERMINAL_COST_MULTIPLIER * (w_track * min_d2 + w_heading * herr * herr);
                int prog = (local_idxs[min_wi] - local_idxs[local_nearest] + (int)waypoints.size()) % (int)waypoints.size();
                double prog_v = (prog > (int)waypoints.size() / 2) ? (double)(waypoints.size() - prog) : -(double)prog;
                b.terminal_cost += w_progress * prog_v;
            }
        }
        return b;
    }

    // =========================================================================
    // [PHẦN 4.12] GHI LOG TELEMETRY & XUẤT FILE CSV
    // =========================================================================
    /**
     * 🎯 CHỨC NĂNG:
     *    In chẩn đoán điều khiển lên console và lưu đủ 25 cột dữ liệu vào file CSV
     *    để công cụ evaluate_track_following.py vẽ đồ thị phân tích.
     */
    void log_diagnostics_and_csv(
            double now_s, double x, double y, double th, double vc,
            double target_v, double final_steer, double final_v,
            const ProfilingTimers& timers, double odom_delay, double lidar_delay,
            const BestTrajectoryBreakdown& best, double min_cost, double w_sum,
            int nearest_wp, double max_c, double min_front_dist, int obs_cnt,
            bool cbf_active, double cbf_v_limit, int wall_cnt, double effective_samples) {
        double cte_x = x - waypoints[nearest_wp].x;
        double cte_y = y - waypoints[nearest_wp].y;
        double cte   = std::hypot(cte_x, cte_y);
        double herr_now = std::abs(normalize_angle(th - waypoint_headings[nearest_wp]));

        double sum_cost = 0.0, max_cost = costs_buf[0];
        for (double c : costs_buf) {
            sum_cost += c;
            if (c > max_cost) max_cost = c;
        }
        double avg_cost = sum_cost / num_samples;

        if (enable_console_log) {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 100,
                "[TRACK DIAG] cte=%.3fm | dx=%.3f | dy=%.3f | herr=%.3frad(%.1f°) | curv=%.2f | tgt_v=%.2f | wp_idx=%d",
                cte, cte_x, cte_y, herr_now, herr_now * 180.0 / M_PI, max_c, target_v, nearest_wp);

            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                "[MPPI COST] Min (Best): %.1f | Avg: %.1f | Max: %.1f | w_sum=%.2e | effective=%.1f (%.1f%%)",
                min_cost, avg_cost, max_cost, w_sum, effective_samples, effective_samples / num_samples * 100.0);

            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                "[MPPI DIAG] Best breakdown: Track: %.1f, Heading: %.1f, Speed: %.1f, Smooth: %.1f, Obs: %.1f, Terminal: %.1f",
                w_track * best.track_cost, w_heading * best.heading_cost, w_speed * best.speed_cost,
                w_smooth * best.smooth_cost, w_obs * best.obs_cost, best.terminal_cost);

            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                "[MPPI DIAG] Collision Rate: %.1f%%", best.collision_rate);

            double steer_diff = std::abs(nominal_control[0].steer - last_ema_steer);
            if (steer_diff > 0.15) {
                RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 500,
                    "[EMA LAG] Vô lăng đang đuổi theo MPPI: MPPI=%.2f rad, EMA=%.2f rad (Lệch: %.2f)", 
                    nominal_control[0].steer, last_ema_steer, steer_diff);
            }

            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
                "[PERF] pure_comp=%.2fms (max CPU: ~%.0fHz) | loop_interval=%.2fms (act: ~%.1fHz) | prep=%.2f | sample=%.2f | cost=%.2f | update=%.2f | shield=%.2f",
                timers.t_total_ms, timers.max_possible_hz, timers.loop_interval_ms, timers.actual_freq_hz,
                timers.t_prep_ms, timers.t_sample_ms, timers.t_cost_ms, timers.t_update_ms, timers.t_shield_ms);
        } else {
            // Heartbeat nhẹ nhàng mỗi 2s: không flood terminal, cung cấp đủ thông tin CPU & tần số
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                "⚡ [MPPI Heartbeat] comp=%.1fms (max CPU: ~%.0fHz) | cur_v=%.2f/tgt=%.2f m/s | cte=%.3fm | cbf=%s | logged",
                timers.t_total_ms, timers.max_possible_hz, vc, target_v, cte, (cbf_active ? "ACTIVE" : "OFF"));
        }

        if (csv_log.is_open()) {
            csv_log << now_s << "," << x << "," << y << "," << th << "," << vc << "," 
                    << target_v << "," << final_steer << "," << final_v << "," 
                    << timers.t_total_ms << ",0,0,0,"
                    << odom_delay << "," << lidar_delay << "," 
                    << w_track * best.track_cost << "," << w_obs * best.obs_cost << "," << best.collision_rate << ","
                    << cte << "," << herr_now << "," << max_c << "," << nominal_control[0].steer << ","
                    << min_cost << "," << avg_cost << "," << min_front_dist << "," << obs_cnt << ","
                    << timers.loop_interval_ms << "," << timers.max_possible_hz << "," << timers.actual_freq_hz << ","
                    << timers.t_prep_ms << "," << timers.t_sample_ms << "," << timers.t_cost_ms << "," << timers.t_update_ms << "," << timers.t_shield_ms << ","
                    << (cbf_active ? 1 : 0) << "," << cbf_v_limit << "," << wall_cnt << ","
                    << w_sum << "," << effective_samples << "\n";
        }

        if (timers.t_total_ms > 50.0) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500,
                "[PERF CẢNH BÁO] Vòng lặp vượt budget 50ms! (%.1fms) — MPPI sẽ chậm hơn 20Hz", timers.t_total_ms);
        }
    }

    // =========================================================================
    // [PHẦN 4.13] ĐIỀU PHỐI VÒNG LẶP CHÍNH (MAIN CONTROL PIPELINE - 20 HZ)
    // =========================================================================
    /**
     * 🎯 CHỨC NĂNG:
     *    Pipeline điều hành chính của bộ điều khiển MPPI, chạy định kỳ ở tần số 20Hz (dt=0.05s).
     * ⚙️ QUY TRÌNH TOÀN DIỆN:
     *    1. Snapshot Pose & LiDAR an toàn đa luồng.
     *    2. Kiểm tra tính toàn vẹn dữ liệu (tránh mất kết nối / NaN).
     *    3. Tìm waypoint gần nhất & trích xuất cửa sổ con 200 điểm.
     *    4. Phân loại chùm tia LiDAR qua Corridor Filter.
     *    5. Tính tốc độ mục tiêu theo độ cong và chướng ngại vật.
     *    6. Sinh mẫu nhiễu Gaussian cho 500 quỹ đạo.
     *    7. Giả lập song song OpenMP 500 mẫu xe & chấm điểm chi phí.
     *    8. Cập nhật quỹ đạo điều khiển danh nghĩa qua Softmax.
     *    9. Làm mượt vô lăng qua bộ lọc EMA.
     *   10. Áp dụng Lá chắn an toàn CBF-QP để chống va chạm.
     *   11. Xuất lệnh lái (/drive) và vẽ quỹ đạo RViz.
     *   12. Tịnh tiến cửa sổ thời gian (Shift Horizon).
     *   13. Tính chi phí mẫu tốt nhất và ghi log CSV.
     */
    void control_loop() {
        auto start_time = std::chrono::high_resolution_clock::now();
        double now_s = now().seconds();

        // Đo khoảng thời gian thực tế giữa 2 chu kỳ gọi control_loop
        double loop_interval_ms = 0.0;
        if (has_last_loop_tp) {
            loop_interval_ms = std::chrono::duration<double, std::milli>(start_time - last_loop_tp).count();
        } else {
            has_last_loop_tp = true;
            loop_interval_ms = dt * 1000.0;
        }
        last_loop_tp = start_time;

        // 1. Thread-safe snapshot pose & LiDAR
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

        double odom_delay = now_s - o_stamp.seconds();
        double lidar_delay = now_s - obs_stamp.seconds();
        if (got_odom && (odom_delay > 0.1 || (obs_stamp.nanoseconds() != 0 && lidar_delay > 0.1))) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500, 
                "[LÀM CHẬM HỆ THỐNG] Độ trễ cảm biến cao! Odom: %.3fs, LiDAR: %.3fs", odom_delay, lidar_delay);
        }

        // 2. Kiểm tra tính hợp lệ dữ liệu
        if (!got_odom || waypoints.empty() || (now_s - o_stamp.seconds() > ODOM_TIMEOUT_S)) {
            last_target_speed = 0.0;
            last_ema_v = 0.0;
            publish_drive(0.0, 0.0);
            return;
        }
        if (std::isnan(x) || std::isnan(y) || std::isnan(th)) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "NaN trong pose!");
            return;
        }

        // 3. Quản lý Waypoint: tìm điểm gần nhất & tạo cửa sổ cục bộ
        int nearest_wp = find_nearest_waypoint(x, y);
        int local_nearest = build_local_waypoint_window(nearest_wp);
        if (local_nearest == -1) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "local_nearest không tìm thấy!");
            return;
        }

        // 4. Lọc hành lang LiDAR (Corridor Filter)
        CorridorResult corridor = filter_lidar_corridor(x, y, th, raw_obs);

        // 5. Tính tốc độ mục tiêu theo độ cong đường đua
        double max_c = 0.0;
        double target_v = compute_target_velocity(nearest_wp, corridor.min_front_dist, max_c);

        // 6. Kiểm tra độ trễ cảm biến LiDAR cho lá chắn CBF
        double obs_age = now_s - obs_stamp.seconds();
        bool obs_fresh = (obs_stamp.nanoseconds() != 0 && obs_age < LIDAR_TIMEOUT_S);
        if (obs_stamp.nanoseconds() != 0 && !obs_fresh) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, 
                "[CẢNH BÁO] LiDAR bị trễ %.2fs (>%.1fs). Bộ lọc an toàn CBF tạm tắt!", obs_age, LIDAR_TIMEOUT_S);
        }

        // 7. Sinh mẫu nhiễu MPPI
        auto t_prep_end = std::chrono::high_resolution_clock::now();
        sample_mppi_noise();
        auto t_sample_end = std::chrono::high_resolution_clock::now();

        // 8. Giả lập song song OpenMP & tính chi phí
        int collision_samples = evaluate_rollouts_parallel(
            x, y, th, target_v, corridor.obs_pts, nearest_wp, local_nearest);
        auto t_cost_end = std::chrono::high_resolution_clock::now();

        // 8b. Bóc tách chi tiết quỹ đạo tốt nhất ngay khi rollout hoàn tất (chuẩn xác 100% trước khi nominal_control bị update/shift)
        int best_idx = std::distance(costs_buf.begin(), std::min_element(costs_buf.begin(), costs_buf.end()));
        double min_cost = costs_buf[best_idx];
        BestTrajectoryBreakdown best = compute_best_trajectory_breakdown(
            best_idx, local_nearest, corridor.obs_pts, target_v, x, y, th, collision_samples);

        // 9. Cập nhật điều khiển danh nghĩa qua Softmax
        double effective_samples = 0.0;
        double w_sum = update_nominal_control(min_cost, target_v, effective_samples);
        auto t_update_end = std::chrono::high_resolution_clock::now();

        // 10. Làm mượt tín hiệu bằng bộ lọc EMA
        Control ema_cmd = apply_ema_filter(nominal_control[0].v, nominal_control[0].steer);

        // 11. Lá chắn an toàn CBF-QP
        bool cbf_active = false;
        double cbf_v_limit = target_v;
        Control safe_cmd = apply_cbf_safety_shield(raw_obs, obs_fresh, ema_cmd.v, ema_cmd.steer, target_v, cbf_active, cbf_v_limit);
        auto t_shield_end = std::chrono::high_resolution_clock::now();

        // 12. Xuất lệnh điều khiển ra xe & vẽ quỹ đạo RViz
        publish_drive(safe_cmd.v, safe_cmd.steer);
        publish_best_trajectory(x, y, th);

        if (enable_console_log) {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 100,
                "MPPI | cost=%6.0f | curv=%.2f | tgt=%.2f | v=%.2f | ema_v=%.2f | ema_s=%5.2f | raw_s=%5.2f | cbf=%s | mode=%s",
                min_cost, max_c, target_v, vc, last_ema_v, last_ema_steer, nominal_control[0].steer,
                (enable_cbf ? "ON" : "OFF"),
                (enable_obstacle_avoidance ? "FULL" : "TRACK_ONLY"));
        }

        // 13. Đo thời gian tính toán thuần túy (pure computation time, trước khi ghi disk & shift)
        auto end_time = std::chrono::high_resolution_clock::now();
        ProfilingTimers timers;
        timers.t_prep_ms        = std::chrono::duration<double, std::milli>(t_prep_end - start_time).count();
        timers.t_sample_ms      = std::chrono::duration<double, std::milli>(t_sample_end - t_prep_end).count();
        timers.t_cost_ms        = std::chrono::duration<double, std::milli>(t_cost_end - t_sample_end).count();
        timers.t_update_ms      = std::chrono::duration<double, std::milli>(t_update_end - t_cost_end).count();
        timers.t_shield_ms      = std::chrono::duration<double, std::milli>(t_shield_end - t_update_end).count();
        timers.t_total_ms       = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        timers.loop_interval_ms = loop_interval_ms;
        timers.max_possible_hz  = (timers.t_total_ms > 0.001) ? (1000.0 / timers.t_total_ms) : 999.0;
        timers.actual_freq_hz   = (loop_interval_ms > 0.001) ? (1000.0 / loop_interval_ms) : (1.0 / dt);

        // Cập nhật thống kê hiệu năng toàn cục
        total_cycles++;
        sum_comp_time_ms += timers.t_total_ms;
        if (timers.t_total_ms < min_comp_time_ms) min_comp_time_ms = timers.t_total_ms;
        if (timers.t_total_ms > max_comp_time_ms) max_comp_time_ms = timers.t_total_ms;
        if (timers.t_total_ms > 50.0) cycles_over_50ms++;
        if (timers.t_total_ms > 33.3) cycles_over_33ms++;
        if (timers.t_total_ms > 25.0) cycles_over_25ms++;
        if (timers.t_total_ms > 20.0) cycles_over_20ms++;

        // 14. Ghi log chẩn đoán và file CSV (chuẩn bị xong dữ liệu chu kỳ hiện tại)
        log_diagnostics_and_csv(
            now_s, x, y, th, vc, target_v, safe_cmd.steer, safe_cmd.v,
            timers, odom_delay, lidar_delay, best, min_cost, w_sum,
            nearest_wp, max_c, corridor.min_front_dist, corridor.obs_cnt,
            cbf_active, cbf_v_limit, corridor.wall_cnt, effective_samples);

        // 15. Tịnh tiến cửa sổ thời gian (Shift Horizon) chuẩn bị cho chu kỳ kế tiếp
        shift_horizon(target_v);
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
        p.x = x; p.y = y; p.z = VISUALIZATION_HEIGHT_Z;
        m.points.push_back(p);

        for (int t = 0; t < horizon; t++) {
            x  += nominal_control[t].v * std::cos(th) * dt;
            y  += nominal_control[t].v * std::sin(th) * dt;
            th += nominal_control[t].v * std::tan(nominal_control[t].steer) / WHEELBASE * dt;
            th  = normalize_angle(th);
            p.x = x; p.y = y; p.z = VISUALIZATION_HEIGHT_Z;
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
    exec.remove_node(node);
    node.reset(); // Huỷ node và in Profiling Report an toàn trước khi tắt ROS context
    rclcpp::shutdown();
    return 0;
}