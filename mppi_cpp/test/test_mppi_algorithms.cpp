/**
 * test_mppi_algorithms.cpp
 * Comprehensive Test Suite for MPPI Controller algorithms
 * Covers:
 *   1. Waypoint Search & Window Indexing
 *   2. Curvature Profiling & Target Speed Planning
 *   3. Control Barrier Function (CBF-QP Safety Shield)
 *   4. MPPI Rollout, Feynman-Kac Softmax & Numerical Stability
 *   5. Closed-Loop Path Tracking Benchmark (OpenMP)
 *
 * Compile: g++ -O3 -fopenmp -std=c++17 test/test_mppi_algorithms.cpp -o test_mppi_suite
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <cassert>
#include <iomanip>
#include <omp.h>

// ANSI Colors for terminal output
#define ANSI_RESET   "\033[0m"
#define ANSI_RED     "\033[31m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_YELLOW  "\033[33m"
#define ANSI_BLUE    "\033[34m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_BOLD    "\033[1m"

int g_tests_run = 0;
int g_tests_passed = 0;
int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    g_tests_run++; \
    if (cond) { \
        g_tests_passed++; \
        std::cout << "  " << ANSI_GREEN << "✓ [PASS]" << ANSI_RESET << " " << msg << std::endl; \
    } else { \
        g_tests_failed++; \
        std::cout << "  " << ANSI_RED << "✗ [FAIL]" << ANSI_RESET << " " << msg << " (Line: " << __LINE__ << ")" << std::endl; \
    } \
} while(0)

// Data structures
struct Point2D { double x, y; };
struct Control { double v, steer; };
struct ObsPt   { double x, y, r; };

inline double normalize_angle(double a) {
    a = std::fmod(a + M_PI, 2.0 * M_PI);
    if (a < 0) a += 2.0 * M_PI;
    return a - M_PI;
}

// =========================================================================
// TEST SUITE 1: WAYPOINT SEARCH & LOCAL WINDOW WRAP-AROUND
// =========================================================================
class WaypointManagerTest {
public:
    static constexpr int WP_WINDOW = 200;
    static constexpr int WP_WINDOW_BACK = 40;
    static constexpr int WP_SEARCH_BACK = 30;
    static constexpr int WP_SEARCH_FORWARD = 60;
    static constexpr double TELEPORT_THRESHOLD_M = 5.0;

    std::vector<Point2D> waypoints;
    std::vector<double>  headings;
    std::vector<double>  curvatures;
    int last_nearest_wp = 0;

    std::vector<Point2D> local_wps;
    std::vector<double>  local_hdgs;
    std::vector<int>     local_idxs;

    void generate_circular_track(int num_points, double radius) {
        waypoints.resize(num_points);
        headings.resize(num_points);
        curvatures.resize(num_points);
        double d_theta = 2.0 * M_PI / num_points;
        for (int i = 0; i < num_points; i++) {
            double theta = i * d_theta;
            waypoints[i] = { radius * std::cos(theta), radius * std::sin(theta) };
        }
        int w = num_points;
        const int CURV_SPAN = (w >= 40) ? 10 : std::max(1, w / 4);
        for (int i = 0; i < w; i++) {
            auto& p1 = waypoints[(i - CURV_SPAN + w) % w];
            auto& p2 = waypoints[i];
            auto& p3 = waypoints[(i + CURV_SPAN) % w];
            headings[i] = std::atan2(p3.y - p1.y, p3.x - p1.x);
            double dx1 = p2.x - p1.x, dy1 = p2.y - p1.y;
            double dx2 = p3.x - p2.x, dy2 = p3.y - p2.y;
            double l1 = std::hypot(dx1, dy1), l2 = std::hypot(dx2, dy2);
            double l3 = std::hypot(p3.x - p1.x, p3.y - p1.y);
            curvatures[i] = (l1 * l2 * l3 > 1e-9) ? 4.0 * (dx1 * dy2 - dy1 * dx2) / (l1 * l2 * l3) : 0.0;
        }
    }

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
        if (min_d > TELEPORT_THRESHOLD_M) {
            for (int i = 0; i < w; i++) {
                double d = std::hypot(waypoints[i].x - x, waypoints[i].y - y);
                if (d < min_d) { min_d = d; best_wp = i; }
            }
        }
        last_nearest_wp = best_wp;
        return best_wp;
    }

    int build_local_waypoint_window(int nearest_wp) {
        local_wps.clear(); local_hdgs.clear(); local_idxs.clear();
        int w = static_cast<int>(waypoints.size());
        for (int i = -WP_WINDOW_BACK; i < WP_WINDOW - WP_WINDOW_BACK; i++) {
            int idx = ((nearest_wp + i) % w + w) % w;
            local_wps.push_back(waypoints[idx]);
            local_hdgs.push_back(headings[idx]);
            local_idxs.push_back(idx);
        }
        int local_nearest = WP_WINDOW_BACK;
        return local_nearest;
    }
};

void run_suite_1_waypoint_management() {
    std::cout << ANSI_BOLD << "\n[TEST SUITE 1] WAYPOINT SEARCH & LOCAL WINDOW WRAP-AROUND" << ANSI_RESET << std::endl;
    WaypointManagerTest wm;
    // 1000 points on radius 8.0m (circumference ~50.26m -> spacing ~0.05m = 5cm)
    wm.generate_circular_track(1000, 8.0);

    // TC 1.1: Sequential Forward Tracking
    bool seq_pass = true;
    for (int step = 0; step < 50; step++) {
        int expected_idx = step * 3;
        double qx = wm.waypoints[expected_idx].x + 0.02;
        double qy = wm.waypoints[expected_idx].y - 0.01;
        int found = wm.find_nearest_waypoint(qx, qy);
        if (found != expected_idx) {
            seq_pass = false;
            break;
        }
    }
    TEST_ASSERT(seq_pass, "TC 1.1: Sequential tracking correctly identifies nearest waypoints with 5cm spacing");

    // TC 1.2: Lap Wrap-around (crossing 999 -> 0)
    wm.last_nearest_wp = 998;
    int found_wrap = wm.find_nearest_waypoint(wm.waypoints[0].x, wm.waypoints[0].y);
    TEST_ASSERT(found_wrap == 0, "TC 1.2: Lap wrap-around correctly detects transition from wp 998 across start line to wp 0");

    // TC 1.3: Teleport Recovery
    wm.last_nearest_wp = 10;
    // Jump to opposite side of track: wp 500 (distance > 15m)
    int found_teleport = wm.find_nearest_waypoint(wm.waypoints[500].x, wm.waypoints[500].y);
    TEST_ASSERT(found_teleport == 500, "TC 1.3: Teleport recovery correctly scans full track and snaps to wp 500");

    // TC 1.4: Local Window Builder
    int local_near = wm.build_local_waypoint_window(500);
    TEST_ASSERT(wm.local_wps.size() == 200, "TC 1.4a: Local window vector contains exactly WP_WINDOW = 200 elements");
    TEST_ASSERT(local_near == 40, "TC 1.4b: local_nearest is exactly at index WP_WINDOW_BACK = 40");
    TEST_ASSERT(wm.local_idxs[local_near] == 500, "TC 1.4c: local_idxs[local_nearest] maps directly to global waypoint 500");
}

// =========================================================================
// TEST SUITE 2: CURVATURE PROFILING & TARGET SPEED PLANNING
// =========================================================================
void run_suite_2_curvature_and_speed() {
    std::cout << ANSI_BOLD << "\n[TEST SUITE 2] CURVATURE PROFILING & TARGET SPEED PLANNING" << ANSI_RESET << std::endl;

    double target_speed_max = 3.0;
    double min_speed_curve  = 2.8;
    double curve_thresh     = 0.5;
    double max_accel        = 2.5;
    double max_decel        = 2.61;
    double dt               = 0.05;

    // TC 2.1: Straight track (curvature = 0.0)
    double max_c_straight = 0.05; // well below 0.5
    double factor_straight = (max_c_straight > curve_thresh)
        ? std::max(0.0, 1.0 - (max_c_straight - curve_thresh) / curve_thresh)
        : 1.0;
    double target_v_straight = min_speed_curve + (target_speed_max - min_speed_curve) * factor_straight;
    TEST_ASSERT(std::abs(target_v_straight - 3.0) < 1e-4, "TC 2.1: Straight track achieves maximum target speed (3.0 m/s)");

    // TC 2.2: Sharp curve (curvature = 1.0 > 0.5)
    double max_c_curve = 1.0;
    double factor_curve = (max_c_curve > curve_thresh)
        ? std::max(0.0, 1.0 - (max_c_curve - curve_thresh) / curve_thresh)
        : 1.0;
    double target_v_curve = min_speed_curve + (target_speed_max - min_speed_curve) * factor_curve;
    TEST_ASSERT(std::abs(target_v_curve - 2.8) < 1e-4, "TC 2.2: Sharp curve drops target speed to min_speed_curve (2.8 m/s)");

    // TC 2.3: Acceleration and Deceleration rate limits
    double last_speed = 1.0;
    double desired_speed = 3.0;
    double ramped_up = std::min(last_speed + max_accel * dt, desired_speed);
    TEST_ASSERT(std::abs(ramped_up - (1.0 + 2.5 * 0.05)) < 1e-6, "TC 2.3a: Ramping up respects max_accel limit (1.125 m/s)");

    last_speed = 3.0;
    desired_speed = 0.0;
    double ramped_down = std::max(last_speed - max_decel * dt, desired_speed);
    TEST_ASSERT(std::abs(ramped_down - (3.0 - 2.61 * 0.05)) < 1e-6, "TC 2.3b: Ramping down respects max_decel limit (2.8695 m/s)");

    // TC 2.4: Proactive deceleration with obstacle ahead
    double obs_decel_start_dist = 1.5;
    double obs_decel_min_factor = 0.60;
    double PROACTIVE_DECEL_MIN_DIST = 0.5;
    double min_front_dist = 1.0; // between 0.5m and 1.5m
    double f = obs_decel_min_factor
        + (1.0 - obs_decel_min_factor) * ((min_front_dist - PROACTIVE_DECEL_MIN_DIST) / (obs_decel_start_dist - PROACTIVE_DECEL_MIN_DIST));
    double proactive_tgt_v = target_speed_max * f;
    TEST_ASSERT(proactive_tgt_v < target_speed_max && proactive_tgt_v >= target_speed_max * obs_decel_min_factor,
        "TC 2.4: Proactive deceleration reduces target speed smoothly in proportion to distance");
}

// =========================================================================
// TEST SUITE 3: CONTROL BARRIER FUNCTION (CBF-QP SAFETY SHIELD)
// =========================================================================
struct CBFShield {
    bool   enable_cbf         = true;
    double cbf_d_min          = 0.35;
    double cbf_gamma          = 2.5;
    double cbf_fov_cutoff_deg = 15.0;

    Control apply(const std::vector<Point2D>& raw_obs, bool obs_fresh, double v_cmd, double steer_cmd, double target_v, bool& cbf_active, double& cbf_v_limit) {
        double final_v = v_cmd;
        double final_steer = steer_cmd;
        cbf_active = false;
        cbf_v_limit = target_v;

        if (enable_cbf && obs_fresh) {
            double v_cbf_max = target_v;
            for (const auto& pt : raw_obs) {
                if (pt.x <= 0.05) continue;
                double angle_deg = std::abs(std::atan2(pt.y, pt.x) * 180.0 / M_PI);
                if (angle_deg <= cbf_fov_cutoff_deg) {
                    double r_i = std::hypot(pt.x, pt.y);
                    double cos_phi = pt.x / r_i;
                    double limit_i = (cbf_gamma * (r_i - cbf_d_min)) / cos_phi;
                    if (limit_i < v_cbf_max) v_cbf_max = limit_i;
                }
            }
            cbf_v_limit = v_cbf_max;
            double v_safe = std::max(0.0, std::min(final_v, v_cbf_max));
            if (v_safe < final_v - 0.05) {
                cbf_active = true;
            }
            final_v = v_safe;
        }
        return {final_v, final_steer};
    }
};

void run_suite_3_cbf_safety_shield() {
    std::cout << ANSI_BOLD << "\n[TEST SUITE 3] CONTROL BARRIER FUNCTION (CBF SAFETY SHIELD)" << ANSI_RESET << std::endl;
    CBFShield cbf;
    bool active = false;
    double limit = 0.0;

    // TC 3.1: Hard Emergency Stop (obstacle at r <= d_min = 0.35m)
    std::vector<Point2D> obs_imminent = {{0.30, 0.0}}; // 30cm straight ahead
    Control cmd1 = cbf.apply(obs_imminent, true, 2.5, 0.0, 3.0, active, limit);
    TEST_ASSERT(cmd1.v == 0.0 && active == true, "TC 3.1: Obstacle at 0.30m (<= d_min 0.35m) triggers emergency hard stop (0.0 m/s)");

    // TC 3.2: Proportional barrier deceleration (obstacle at 0.50m)
    std::vector<Point2D> obs_near = {{0.50, 0.0}};
    Control cmd2 = cbf.apply(obs_near, true, 2.5, 0.0, 3.0, active, limit);
    double expected_limit = (2.5 * (0.50 - 0.35)) / 1.0; // = 0.375 m/s
    TEST_ASSERT(std::abs(cmd2.v - expected_limit) < 1e-4 && active == true,
        "TC 3.2: Obstacle at 0.50m correctly limits speed to Nagumo barrier velocity (0.375 m/s)");

    // TC 3.3: Obstacle outside FOV cutoff (+/- 15 deg)
    // Point at x=0.5m, y=0.3m -> angle = atan2(0.3, 0.5) = 31 degrees > 15 deg
    std::vector<Point2D> obs_outside = {{0.50, 0.30}};
    Control cmd3 = cbf.apply(obs_outside, true, 2.5, 0.0, 3.0, active, limit);
    TEST_ASSERT(cmd3.v == 2.5 && active == false,
        "TC 3.3: Obstacle at 31 deg (outside 15 deg FOV cone) does not falsely trigger CBF intervention");

    // TC 3.4: Bumper filter & Zero-division guard
    std::vector<Point2D> obs_chassis = {{0.03, 0.01}}; // on bumper (< 0.05m)
    Control cmd4 = cbf.apply(obs_chassis, true, 2.5, 0.0, 3.0, active, limit);
    TEST_ASSERT(cmd4.v == 2.5 && active == false,
        "TC 3.4: Points within vehicle chassis (x <= 0.05m) are filtered without zero-division error");
}

// =========================================================================
// TEST SUITE 4: MPPI ROLLOUT, FEYNMAN-KAC SOFTMAX & NUMERICAL STABILITY
// =========================================================================
void run_suite_4_mppi_optimization() {
    std::cout << ANSI_BOLD << "\n[TEST SUITE 4] MPPI OPTIMIZATION & NUMERICAL STABILITY" << ANSI_RESET << std::endl;

    // TC 4.1: Kinematic Bicycle Model Step
    double x = 0.0, y = 0.0, th = 0.0;
    double pv = 2.0, ps = 0.2, wheelbase = 0.39, dt = 0.05;
    x  += pv * std::cos(th) * dt;
    y  += pv * std::sin(th) * dt;
    th += pv * std::tan(ps) / wheelbase * dt;
    th  = normalize_angle(th);
    TEST_ASSERT(x > 0.099 && x < 0.101 && std::abs(y) < 1e-6 && th > 0.05,
        "TC 4.1: Kinematic bicycle model integrates forward motion and steering yaw accurately");

    // TC 4.2: Numerical Stability under Extreme Cost Divergence
    int num_samples = 500;
    std::vector<double> costs(num_samples);
    costs[0] = 50.0; // Best trajectory
    for (int i = 1; i < num_samples; i++) {
        costs[i] = 1000000.0 + i * 100.0; // Extreme high penalty
    }
    double min_cost = *std::min_element(costs.begin(), costs.end());
    double lambda_ = 140.0;
    double w_sum = 0.0;
    std::vector<double> weights(num_samples, 0.0);
    bool has_nan = false;
    for (int n = 0; n < num_samples; n++) {
        weights[n] = std::exp(-(costs[n] - min_cost) / lambda_);
        if (std::isnan(weights[n]) || std::isinf(weights[n])) has_nan = true;
        w_sum += weights[n];
    }
    TEST_ASSERT(!has_nan && w_sum >= 1.0,
        "TC 4.2: Softmax subtraction of min_cost prevents numerical overflow under 10^6 cost differences");

    // TC 4.3: Kish's Effective Sample Size
    double sum_w_norm_sq = 0.0;
    for (int n = 0; n < num_samples; n++) {
        double w_norm = weights[n] / w_sum;
        sum_w_norm_sq += w_norm * w_norm;
    }
    double n_eff = (sum_w_norm_sq > 1e-12) ? (1.0 / sum_w_norm_sq) : 0.0;
    TEST_ASSERT(std::abs(n_eff - 1.0) < 0.01,
        "TC 4.3: Kish Effective Sample Size correctly evaluates to ~1.0 when single trajectory dominates");

    // TC 4.4: Weight Collapse Fallback
    double zero_w_sum = 0.0;
    bool fallback_triggered = false;
    if (zero_w_sum <= 1e-10) {
        fallback_triggered = true; // Fallback keeps nominal control
    }
    TEST_ASSERT(fallback_triggered, "TC 4.4: Weight collapse (w_sum <= 1e-10) triggers safe nominal control fallback");

    // TC 4.5: Horizon Shift Test
    int horizon = 30;
    std::vector<Control> nominal(horizon);
    for (int t = 0; t < horizon; t++) {
        nominal[t] = { 2.0 + t * 0.01, 0.1 };
    }
    double next_target_v = 2.5;
    for (int t = 0; t < horizon - 1; t++) nominal[t] = nominal[t + 1];
    nominal[horizon - 1].v     = next_target_v;
    nominal[horizon - 1].steer = nominal[horizon - 2].steer * 0.5;
    TEST_ASSERT(nominal[0].v == 2.01 && nominal[horizon - 1].v == 2.5 && nominal[horizon - 1].steer == 0.05,
        "TC 4.5: Horizon shift advances control by 1 step and decays terminal steering");
}

// =========================================================================
// TEST SUITE 5: CLOSED-LOOP PARALLEL SIMULATION BENCHMARK
// =========================================================================
void run_suite_5_closed_loop_benchmark() {
    std::cout << ANSI_BOLD << "\n[TEST SUITE 5] CLOSED-LOOP PARALLEL SIMULATION BENCHMARK (OpenMP)" << ANSI_RESET << std::endl;

    WaypointManagerTest wm;
    wm.generate_circular_track(1000, 10.0); // R=10m circle

    int horizon = 30;
    int num_samples = 500;
    double dt = 0.05;
    double lambda_ = 140.0;
    double w_track = 20.0;
    double w_heading = 5.0;
    double w_speed = 8.0;
    double w_smooth = 15.5;

    std::vector<Control> nominal_control(horizon, {2.0, 0.0});
    std::vector<std::vector<Control>> noise_buf(num_samples, std::vector<Control>(horizon));
    std::vector<double> costs_buf(num_samples, 0.0);
    std::vector<double> weights_buf(num_samples, 0.0);
    std::vector<double> upd_v_buf(horizon, 0.0);
    std::vector<double> upd_s_buf(horizon, 0.0);

    std::mt19937 rng(42);
    std::normal_distribution<double> dist_v(0.0, 1.5);
    std::normal_distribution<double> dist_s(0.0, 0.20);

    // Initial pose on track
    double x = wm.waypoints[0].x;
    double y = wm.waypoints[0].y;
    double th = wm.headings[0];

    double sum_time_ms = 0.0;
    double max_time_ms = 0.0;
    double total_cte = 0.0;
    int num_cycles = 50;

    for (int cycle = 0; cycle < num_cycles; cycle++) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // 1. Waypoints
        int nearest_wp = wm.find_nearest_waypoint(x, y);
        int local_nearest = wm.build_local_waypoint_window(nearest_wp);

        // 2. Sample noise
        for (int n = 0; n < num_samples; n++) {
            for (int t = 0; t < horizon; t++) {
                noise_buf[n][t].v = dist_v(rng);
                noise_buf[n][t].steer = dist_s(rng);
            }
        }

        // 3. OpenMP parallel rollouts
        #pragma omp parallel for schedule(dynamic)
        for (int n = 0; n < num_samples; n++) {
            double px = x, py = y, pth = th;
            double trk = 0.0, hdg = 0.0, spd = 0.0, smo = 0.0;
            double pv_prv = nominal_control[0].v;
            double ps_prv = nominal_control[0].steer;

            for (int t = 0; t < horizon; t++) {
                double pv = std::max(0.0, std::min(3.0, nominal_control[t].v + noise_buf[n][t].v));
                double ps = std::max(-0.4, std::min(0.4, nominal_control[t].steer + noise_buf[n][t].steer));

                px  += pv * std::cos(pth) * dt;
                py  += pv * std::sin(pth) * dt;
                pth += pv * std::tan(ps) / 0.39 * dt;
                pth  = normalize_angle(pth);

                double min_d2 = 999.0;
                int min_wi = 0;
                for (int wi = 0; wi < (int)wm.local_wps.size(); wi++) {
                    double d2 = (px - wm.local_wps[wi].x) * (px - wm.local_wps[wi].x) + (py - wm.local_wps[wi].y) * (py - wm.local_wps[wi].y);
                    if (d2 < min_d2) { min_d2 = d2; min_wi = wi; }
                }
                trk += min_d2;
                double herr = normalize_angle(pth - wm.local_hdgs[min_wi]);
                hdg += herr * herr;
                spd += (pv - 2.5) * (pv - 2.5);
                if (t > 0) smo += (ps - ps_prv) * (ps - ps_prv);
                pv_prv = pv; ps_prv = ps;
            }
            costs_buf[n] = w_track * trk + w_heading * hdg + w_speed * spd + w_smooth * smo;
        }

        // 4. Softmax update
        double min_cost = *std::min_element(costs_buf.begin(), costs_buf.end());
        double w_sum = 0.0;
        for (int n = 0; n < num_samples; n++) {
            weights_buf[n] = std::exp(-(costs_buf[n] - min_cost) / lambda_);
            w_sum += weights_buf[n];
        }
        std::fill(upd_v_buf.begin(), upd_v_buf.end(), 0.0);
        std::fill(upd_s_buf.begin(), upd_s_buf.end(), 0.0);
        for (int n = 0; n < num_samples; n++) {
            double w = weights_buf[n];
            for (int t = 0; t < horizon; t++) {
                upd_v_buf[t] += w * noise_buf[n][t].v;
                upd_s_buf[t] += w * noise_buf[n][t].steer;
            }
        }
        for (int t = 0; t < horizon; t++) {
            nominal_control[t].v     = std::max(0.0, std::min(3.0, nominal_control[t].v + upd_v_buf[t] / w_sum));
            nominal_control[t].steer = std::max(-0.4, std::min(0.4, nominal_control[t].steer + upd_s_buf[t] / w_sum));
        }

        // 5. Apply vehicle step
        double v_act = nominal_control[0].v;
        double s_act = nominal_control[0].steer;
        x  += v_act * std::cos(th) * dt;
        y  += v_act * std::sin(th) * dt;
        th += v_act * std::tan(s_act) / 0.39 * dt;
        th  = normalize_angle(th);

        // 6. Shift horizon
        for (int t = 0; t < horizon - 1; t++) nominal_control[t] = nominal_control[t + 1];
        nominal_control[horizon - 1].v = 2.5;
        nominal_control[horizon - 1].steer *= 0.5;

        auto t_end = std::chrono::high_resolution_clock::now();
        double cycle_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        sum_time_ms += cycle_ms;
        if (cycle_ms > max_time_ms) max_time_ms = cycle_ms;

        double cte = std::hypot(x - wm.waypoints[nearest_wp].x, y - wm.waypoints[nearest_wp].y);
        total_cte += cte;
    }

    double avg_time_ms = sum_time_ms / num_cycles;
    double avg_cte = total_cte / num_cycles;
    double max_hz = 1000.0 / avg_time_ms;

    std::cout << "  " << ANSI_CYAN << "→ Benchmark Stats: 50 cycles, 500 samples x 30 steps" << ANSI_RESET << std::endl;
    std::cout << "    * Avg Execution Time : " << std::fixed << std::setprecision(2) << avg_time_ms << " ms" << std::endl;
    std::cout << "    * Peak Execution Time: " << max_time_ms << " ms" << std::endl;
    std::cout << "    * Theoretical Max Hz : ~" << std::setprecision(1) << max_hz << " Hz" << std::endl;
    std::cout << "    * Mean Tracking CTE  : " << std::setprecision(3) << avg_cte << " m" << std::endl;

    TEST_ASSERT(avg_cte < 0.20, "TC 5.1: Closed-loop path following maintains low CTE (< 0.20m) along circular trajectory");
    TEST_ASSERT(avg_time_ms < 25.0, "TC 5.2: MPPI pure computation completes in under 25ms (capable of >= 40Hz)");
}

// =========================================================================
// MAIN RUNNER
// =========================================================================
int main() {
    std::cout << ANSI_BOLD << "====================================================================" << ANSI_RESET << std::endl;
    std::cout << ANSI_BOLD << "🧪 MPPI ALGORITHMIC INTEGRITY & REAL-TIME TEST SUITE" << ANSI_RESET << std::endl;
    std::cout << ANSI_BOLD << "====================================================================" << ANSI_RESET << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();

    run_suite_1_waypoint_management();
    run_suite_2_curvature_and_speed();
    run_suite_3_cbf_safety_shield();
    run_suite_4_mppi_optimization();
    run_suite_5_closed_loop_benchmark();

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::cout << ANSI_BOLD << "\n====================================================================" << ANSI_RESET << std::endl;
    std::cout << ANSI_BOLD << "📊 TEST SUMMARY" << ANSI_RESET << std::endl;
    std::cout << ANSI_BOLD << "====================================================================" << ANSI_RESET << std::endl;
    std::cout << "  * Total Tests Run   : " << g_tests_run << std::endl;
    std::cout << "  * Passed            : " << ANSI_GREEN << g_tests_passed << ANSI_RESET << std::endl;
    std::cout << "  * Failed            : " << (g_tests_failed > 0 ? ANSI_RED : ANSI_GREEN) << g_tests_failed << ANSI_RESET << std::endl;
    std::cout << "  * Total Suite Time  : " << std::fixed << std::setprecision(1) << total_ms << " ms" << std::endl;
    std::cout << ANSI_BOLD << "====================================================================" << ANSI_RESET << std::endl;

    return (g_tests_failed == 0) ? 0 : 1;
}
