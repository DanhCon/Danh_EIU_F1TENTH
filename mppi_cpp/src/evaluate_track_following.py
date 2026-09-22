#!/usr/bin/env python3
"""
evaluate_track_following.py — Đánh giá độ bám đường của xe F1TENTH
===================================================================
Input : CSV log do mppi_xe.cpp ghi (~/mppi_logs/mppi_log_*.csv)
        + file waypoint raceline (nếu có) để tính cross-track error chuẩn
Output: Báo cáo điểm số (console + .txt) và biểu đồ phân tích (.png)

Cách dùng:
    python3 evaluate_track_following.py --csv ~/mppi_logs/mppi_log_xxx.csv
    python3 evaluate_track_following.py --csv log.csv --waypoints raceline.csv --output ket_qua

Cột CSV cần có (từ mppi_xe.cpp):
    time,x,y,theta,v_cur,target_v,steer_cmd,v_cmd,exec_time_ms,
    is_stopped,is_stuck,front_blocked,odom_delay,lidar_delay,
    best_track_cost,best_obs_cost,collision_rate,cte,heading_err,
    curvature,steer_raw,min_cost,avg_cost,min_front_dist,obs_cnt
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_WAYPOINT = "/home/fablab_01/danh_pp_ws/install/waypoint/share/waypoint/" \
                   "f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"

# ============================================================
# Đọc dữ liệu
# ============================================================
def load_waypoints(path):
    """Đọc waypoint CSV (x,y mỗi dòng, bỏ dòng #/rỗng)."""
    wps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                wps.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if len(wps) < 3:
        raise RuntimeError(f"File waypoint không hợp lệ: {path} (chỉ có {len(wps)} điểm)")
    return np.array(wps)


def load_log(path):
    """Đọc CSV log, trả về dict {tên cột: np.ndarray}."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"File log rỗng: {path}")
    cols = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k, v in r.items():
            try:
                cols[k].append(float(v))
            except (ValueError, TypeError):
                cols[k].append(math.nan)
    return {k: np.array(v) for k, v in cols.items()}


def has_col(d, name):
    return name in d and len(d[name]) > 0


# ============================================================
# Tính cross-track error chuẩn (vuông góc với đoạn raceline)
# ============================================================
def signed_cte(x, y, wps, hint=0):
    """
    CTE có dấu: khoảng cách vuông góc từ xe tới đoạn waypoint gần nhất.
    Dương = xe lệch TRÁI so với hướng chạy, Âm = lệch PHẢI.
    Trả về (cte, idx_nearest, arc_len).
    """
    n = len(wps)
    cte = np.empty(len(x))
    idx = np.empty(len(x), dtype=int)
    arc = np.empty(len(x))
    last = hint
    cum = 0.0
    seg_len = np.hypot(np.diff(wps[:, 0]), np.diff(wps[:, 1]))
    for i in range(len(x)):
        # tìm waypoint gần nhất trong cửa sổ quanh vị trí trước
        best_d = 1e18
        best_k = last
        for k in range(last - 40, last + 40):
            kk = k % n
            d = (wps[kk, 0] - x[i]) ** 2 + (wps[kk, 1] - y[i]) ** 2
            if d < best_d:
                best_d = d
                best_k = kk
        last = best_k
        idx[i] = best_k

        # chiếu lên đoạn [wp_k, wp_{k+1}]
        p1 = wps[best_k]
        p2 = wps[(best_k + 1) % n]
        dx, dy = p2 - p1
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            cte[i] = math.nan
            arc[i] = cum
            continue
        t = ((x[i] - p1[0]) * dx + (y[i] - p1[1]) * dy) / L2
        t = max(0.0, min(1.0, t))
        px, py = p1[0] + t * dx, p1[1] + t * dy
        dist = math.hypot(x[i] - px, y[i] - py)
        cross = dx * (y[i] - p1[1]) - dy * (x[i] - p1[0])
        cte[i] = dist if cross >= 0 else -dist
        arc[i] = cum + t * math.hypot(dx, dy)
        if i > 0:
            cum += math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
    return cte, idx, arc


def track_heading_at(wps, idx):
    """Hướng của đoạn waypoint tại idx (rad)."""
    n = len(wps)
    dx = wps[(idx + 1) % n, 0] - wps[idx, 0]
    dy = wps[(idx + 1) % n, 1] - wps[idx, 1]
    return np.arctan2(dy, dx)


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ============================================================
# Tính điểm số
# ============================================================
def percentile(x, q):
    return float(np.nanpercentile(x, q)) if len(x) else 0.0


def report_score(cte, herr, verr, flips, steer_std, n_stops, fraction_stopped,
                 target_speed_max):
    """
    Điểm tổng 0-100 dựa trên:
      CTE (40đ) | Heading (20đ) | Speed (15đ) | Mượt lái (15đ) | Dừng (10đ)
    """
    rms_cte = float(np.sqrt(np.nanmean(cte ** 2))) if len(cte) else 99.0
    rms_herr = float(np.sqrt(np.nanmean(herr ** 2))) if len(herr) else 99.0
    rms_verr = float(np.sqrt(np.nanmean(verr ** 2))) if len(verr) else 99.0

    # CTE: lý tưởng < 0.10m, kém khi > 0.40m
    s_cte = 40.0 * max(0.0, min(1.0, 1.0 - (rms_cte - 0.10) / 0.30))
    # Heading: lý tưởng < 0.10 rad, kém khi > 0.35 rad
    s_herr = 20.0 * max(0.0, min(1.0, 1.0 - (rms_herr - 0.10) / 0.25))
    # Tốc độ: sai lệch ≤ 10% target là tốt
    ref_v = max(0.5, target_speed_max)
    s_verr = 15.0 * max(0.0, min(1.0, 1.0 - rms_verr / (0.20 * ref_v)))
    # Mượt lái: dựa trên độ lệch chuẩn đạo hàm lái
    s_smo = 15.0 * max(0.0, min(1.0, 1.0 - steer_std / 1.5)) if steer_std is not None else 10.0
    # Dừng: mỗi lần dừng (không do escape lỗi) trừ 2đ, tối đa trừ 10đ
    s_stop = 10.0 * max(0.0, 1.0 - 0.2 * n_stops) * (1.0 - fraction_stopped)

    total = s_cte + s_herr + s_verr + s_smo + s_stop
    if total >= 85:
        verdict = "🏆 XUẤT SẮC — bám đường rất tốt"
    elif total >= 70:
        verdict = "✅ TỐT — chấp nhận được, có thể tăng tốc"
    elif total >= 55:
        verdict = "⚠️ KHÁ — cần tinh chỉnh thêm"
    elif total >= 40:
        verdict = "🔧 TRUNG BÌNH — xe lệch nhiều, cần tuning"
    else:
        verdict = "❌ KÉM — kiểm tra localization / raceline / trọng số"
    return total, dict(s_cte=s_cte, s_herr=s_herr, s_verr=s_verr,
                       s_smo=s_smo, s_stop=s_stop), verdict


# ============================================================
# Vẽ biểu đồ
# ============================================================
def make_plots(out_png, x, y, theta, t, v_cur, target_v, steer_cmd,
               cte, herr, wps, exec_time, exec_time_limit=50.0):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.25)

    # (1) Đường đua + quỹ đạo xe
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(wps[:, 0], wps[:, 1], "k-", lw=1.0, alpha=0.7, label="Raceline")
    ax.plot(wps[:, 0], wps[:, 1], "k.", ms=2, alpha=0.5)
    ax.plot(x, y, color="#2563eb", lw=1.2, label="Quỹ đạo xe")
    ax.scatter(x[0], y[0], c="green", s=60, marker="o", zorder=5, label="Start")
    ax.scatter(x[-1], y[-1], c="red", s=60, marker="x", zorder=5, label="End")
    ax.set_title("Quỹ đạo xe vs Raceline")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal")

    # (2) Cross-track error theo quãng đường
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(arc, cte, color="#ef4444", lw=0.8, label="CTE (m)")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axhspan(-0.15, 0.15, color="green", alpha=0.12, label="±0.15m (tốt)")
    ax.axhspan(-0.30, 0.30, color="orange", alpha=0.08, label="±0.30m (ok)")
    ax.fill_between(arc, cte, 0, where=(cte > 0), color="red", alpha=0.15)
    ax.set_title("Cross-track error theo quãng đường")
    ax.set_xlabel("Quãng đường đi (m)")
    ax.set_ylabel("CTE (m) — (+) trái, (−) phải")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # (3) Heading error
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, herr * 180.0 / math.pi, color="#8b5cf6", lw=0.8,
            label="Heading err (độ)")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_title("Heading error theo thời gian")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Lệch hướng (độ)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # (4) Tốc độ
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, v_cur, color="#2563eb", lw=0.9, label="v_cur (thực tế)")
    ax.plot(t, target_v, color="#f59e0b", lw=0.9, ls="--", label="target_v (mục tiêu)")
    ax.set_title("Tốc độ thực tế vs mục tiêu")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("v (m/s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # (5) Lệnh lái
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, steer_cmd * 180.0 / math.pi, color="#10b981", lw=0.8,
            label="steer_cmd (độ)")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_title("Lệnh lái theo thời gian (nhiều gai = rack-rack)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Góc lái (độ)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # (6) Histogram CTE
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(cte, bins=60, color="#3b82f6", alpha=0.75)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.axvspan(-0.15, 0.15, color="green", alpha=0.15)
    ax.set_title("Phân bố Cross-track error")
    ax.set_xlabel("CTE (m)")
    ax.set_ylabel("Số mẫu")
    ax.grid(alpha=0.3)

    # Perf nhỏ bên góc phải trên biểu đồ đầu
    ax2 = fig.add_subplot(gs[0, 0], sharex=ax)
    ax2.remove()

    fig.suptitle(f"ĐÁNH GIÁ ĐỘ BÁM ĐƯỜNG — {os.path.basename(out_png)}",
                 fontsize=14, y=0.98)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Đánh giá độ bám đường xe F1TENTH từ CSV log")
    ap.add_argument("--csv", required=True, help="Đường dẫn file CSV log của mppi_xe.cpp")
    ap.add_argument("--waypoints", default=DEFAULT_WAYPOINT,
                    help="Đường dẫn file waypoint raceline (nếu không có sẽ dùng cột cte trong log)")
    ap.add_argument("--output", default=None, help="Tên file đầu ra (không đuôi), mặc định: <csv>_eval")
    args = ap.parse_args()

    base = args.output or os.path.splitext(os.path.basename(args.csv))[0] + "_eval"
    out_png = base + ".png"
    out_txt = base + ".txt"

    print(f"Đọc log: {args.csv}")
    d = load_log(args.csv)
    t = d.get("time", np.arange(len(d.get("x", []))))
    x, y = d["x"], d["y"]
    theta = d.get("theta", np.zeros_like(x))
    v_cur = d.get("v_cur", np.zeros_like(x))
    target_v = d.get("target_v", v_cur.copy())
    steer_cmd = d.get("steer_cmd", np.zeros_like(x))
    exec_time = d.get("exec_time_ms", np.zeros_like(x))
    is_stopped = d.get("is_stopped", np.zeros_like(x))

    # ---- CTE & heading error ----
    wps = None
    try:
        if os.path.exists(args.waypoints):
            wps = load_waypoints(args.waypoints)
            print(f"Waypoints: {len(wps)} điểm ({args.waypoints})")
    except Exception as e:
        print(f"⚠️ Không đọc được waypoint ({e}) — dùng cột cte trong log")

    if wps is not None:
        cte, idx_wp, arc = signed_cte(x, y, wps)
        h_track = track_heading_at(wps, idx_wp)
        herr = np.abs(np.array([wrap_angle(th - h) for th, h in zip(theta, h_track)]))
    else:
        cte = d.get("cte", np.zeros_like(x))
        if has_col(d, "heading_err"):
            herr = np.abs(d["heading_err"])
        else:
            herr = np.zeros_like(x)
        arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))

    verr = v_cur - target_v

    # ---- Thống kê ----
    mask_run = is_stopped < 0.5
    cte_run = cte[mask_run] if np.any(mask_run) else cte
    herr_run = herr[mask_run] if np.any(mask_run) else herr
    verr_run = verr[mask_run] if np.any(mask_run) else verr

    rms_cte = float(np.sqrt(np.nanmean(cte_run ** 2)))
    mean_cte = float(np.nanmean(cte_run))
    p95_cte = percentile(cte_run, 95)
    max_cte = float(np.nanmax(np.abs(cte_run)))
    frac_good = float(np.nanmean(np.abs(cte_run) < 0.15) * 100.0)
    frac_ok = float(np.nanmean(np.abs(cte_run) < 0.30) * 100.0)
    frac_bad = float(np.nanmean(np.abs(cte_run) > 0.50) * 100.0)

    rms_herr = float(np.sqrt(np.nanmean(herr_run ** 2)))
    mean_herr = float(np.nanmean(herr_run) * 180.0 / math.pi)

    # Lệnh lái: số lần đổi chiều (flip) + độ lệch chuẩn đạo hàm
    if len(steer_cmd) > 2:
        s = steer_cmd
        flip = int(np.sum(np.sign(s[1:]) != np.sign(s[:-1])))
        dsteer = np.diff(s) / np.maximum(np.diff(t), 1e-6)
        steer_std = float(np.nanstd(dsteer))
    else:
        flip, steer_std = 0, 0.0
    flip_per_100m = flip / max(arc[-1] / 100.0, 1e-6) if len(arc) else 0.0

    # Dừng xe
    n_stops = int(np.sum((is_stopped[1:] > 0.5) & (is_stopped[:-1] < 0.5)))
    fraction_stopped = float(np.mean(is_stopped > 0.5)) if len(is_stopped) else 0.0

    # Quãng đường & tốc độ
    total_dist = float(arc[-1]) if len(arc) else 0.0
    n_laps = int(total_dist / (np.sum(np.hypot(np.diff(wps[:, 0]), np.diff(wps[:, 1])))
                               if wps is not None else 100.0))
    mean_v = float(np.nanmean(v_cur))
    max_v = float(np.nanmax(v_cur))
    rms_verr = float(np.sqrt(np.nanmean(verr_run ** 2)))

    # Perf
    mean_exec = float(np.nanmean(exec_time))
    max_exec = float(np.nanmax(exec_time))
    over_budget = float(np.nanmean(exec_time > 50.0) * 100.0)

    # ---- Điểm ----
    target_speed_max = float(np.nanmax(target_v)) if len(target_v) else 1.5
    total, parts, verdict = report_score(
        cte_run, herr_run, verr_run, flip, steer_std, n_stops,
        fraction_stopped, target_speed_max)

    # ---- Báo cáo ----
    lines = []
    lines.append("=" * 62)
    lines.append(f"  ĐÁNH GIÁ ĐỘ BÁM ĐƯỜNG — {os.path.basename(args.csv)}")
    lines.append("=" * 62)
    lines.append(f"Thời gian chạy        : {t[-1] - t[0]:7.1f} s")
    lines.append(f"Quãng đường đi        : {total_dist:7.1f} m")
    if wps is not None:
        lines.append(f"Số vòng hoàn thành    : {n_laps}")
    lines.append(f"Tốc độ trung bình/max  : {mean_v:.2f} / {max_v:.2f} m/s")
    lines.append("-" * 62)
    lines.append("  CROSS-TRACK ERROR (CTE) — độ lệch khỏi raceline")
    lines.append(f"  Mean CTE              : {mean_cte:+.3f} m")
    lines.append(f"  RMS CTE               : {rms_cte:.3f} m")
    lines.append(f"  P95 CTE               : {p95_cte:+.3f} m")
    lines.append(f"  Max |CTE|             : {max_cte:.3f} m")
    lines.append(f"  Trong ±0.15m          : {frac_good:5.1f} %")
    lines.append(f"  Trong ±0.30m          : {frac_ok:5.1f} %")
    lines.append(f"  Ngoài ±0.50m          : {frac_bad:5.1f} %")
    lines.append("-" * 62)
    lines.append("  HEADING & TỐC ĐỘ")
    lines.append(f"  Mean heading error    : {mean_herr:6.1f} °")
    lines.append(f"  RMS heading error     : {rms_herr * 180.0 / math.pi:6.1f} °")
    lines.append(f"  RMS sai lệch tốc độ   : {rms_verr:.3f} m/s")
    lines.append("-" * 62)
    lines.append("  ĐỘ MƯỢT LÁI")
    lines.append(f"  Số lần đổi chiều lái  : {flip}  ({flip_per_100m:.1f} lần/100m)")
    lines.append(f"  Độ lệch chuẩn dsteer  : {steer_std:.3f} rad/s  "
                 f"({'rung mạnh' if steer_std > 1.5 else ('hơi rung' if steer_std > 0.8 else 'êm' )})")
    lines.append("-" * 62)
    lines.append("  AN TOÀN & HIỆU NĂNG")
    lines.append(f"  Số lần dừng (escape)  : {n_stops}")
    lines.append(f"  Thời gian dừng        : {fraction_stopped * 100.0:.1f} %")
    lines.append(f"  exec_time mean/max    : {mean_exec:.2f} / {max_exec:.2f} ms")
    lines.append(f"  Vượt budget 50ms      : {over_budget:.1f} %")
    lines.append("-" * 62)
    lines.append("  ĐIỂM SỐ (0-100)")
    lines.append(f"  Bám đường (40)  : {parts['s_cte']:5.1f}")
    lines.append(f"  Hướng (20)      : {parts['s_herr']:5.1f}")
    lines.append(f"  Tốc độ (15)     : {parts['s_verr']:5.1f}")
    lines.append(f"  Mượt lái (15)   : {parts['s_smo']:5.1f}")
    lines.append(f"  An toàn (10)    : {parts['s_stop']:5.1f}")
    lines.append("  " + "-" * 30)
    lines.append(f"  TỔNG            : {total:5.1f} / 100")
    lines.append(f"  KẾT LUẬN        : {verdict}")
    lines.append("=" * 62)

    report = "\n".join(lines)
    print(report)
    with open(out_txt, "w") as f:
        f.write(report + "\n")
    print(f"Đã lưu báo cáo: {out_txt}")

    # ---- Biểu đồ ----
    if wps is not None:
        make_plots(out_png, x, y, theta, t, v_cur, target_v, steer_cmd,
                   cte, herr, wps, exec_time)
    else:
        make_plots(out_png, x, y, theta, t, v_cur, target_v, steer_cmd,
                   cte, herr, np.column_stack([x, y]), exec_time)
    print(f"Đã lưu biểu đồ  : {out_png}")


if __name__ == "__main__":
    main()