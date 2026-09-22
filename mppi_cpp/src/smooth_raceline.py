#!/usr/bin/env python3
"""
smooth_raceline.py — Làm mượt + tái mẫu đều raceline (vòng kín) cho xe F1TENTH
=================================================================================

VẤN ĐỀ GIẢI QUYẾT
------------------
1. Spacing waypoint không đều (3cm → 36cm): code C++ tính curvature bằng
   5-point span theo INDEX, nên vùng thưa bị "phóng đại span", vùng dày bị
   "phóng đại nhiễu" → curvature giả khổng lồ (3+ /m) dù đường thật dễ chạy.
2. Raceline tự vẽ/tay bấm bị zig-zag nhiễu → cua giả (bán kính 0.2-0.4m)
   mà xe (min radius ~0.73m, max curvature ~1.37/m) không thể bám.
3. Nhiều cua gắt thật sự (track nhỏ ~21m): cần "cắt cua" để raceline bám được.

CÁCH DÙNG
---------
    python3 smooth_raceline.py                          # chạy mặc định
    python3 smooth_raceline.py --check-only            # chỉ kiểm tra file, không sửa
    python3 smooth_raceline.py --input duong.csv --output duong_smooth.csv \
            --spacing 0.2 --max-curv 1.0 --window 2.5  # tùy chỉnh

THUẬT TOÁN
----------
1. Resample đều theo arc-length (closed loop) → spacing nhất quán.
2. Rolling average (window theo mét, vòng kín) → xóa nhiễu + cắt cua gắt.
   Chế độ --window=0: TỰ TÌM window tăng dần 0.3→4.0m cho tới khi
   curvature (đo đúng công thức C++) ≤ --max-curv, hoặc hết cỡ.
3. Resample lần cuối về --spacing, tính heading bằng công thức C++.
4. Báo cáo trước/sau: spacing, curvature p50/p95/p99/max, % điểm > 1.0/m
   và > 1.5/m, độ lệch so với raceline gốc, số điểm.
"""

import argparse
import sys

import numpy as np

CSV_HEADER = "x,y,heading"


# ---------------------------------------------------------------- I/O
def load_waypoints(path: str) -> np.ndarray:
    """Đọc CSV x,y[,heading] (có thể có header dòng 1)."""
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    except Exception as e:
        try:
            data = np.loadtxt(path, delimiter=",")
        except Exception:
            raise ValueError(f"Không đọc được file: {path} ({e})")
    if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 10:
        raise ValueError(f"File {path} phải là CSV 2 cột x,y, tối thiểu 10 điểm "
                         f"(tìm thấy shape={getattr(data, 'shape', None)})")
    return data[:, :2]


def save_waypoints(path: str, pts: np.ndarray, headings: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(CSV_HEADER + "\n")
        for (x, y), h in zip(pts, headings):
            f.write(f"{x:.6f},{y:.6f},{h:.6f}\n")


# -------------------------------------------------------- hình học
def resample_uniform(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Resample vòng kín theo arc-length với spacing cố định."""
    s = np.zeros(len(pts))
    s[1:] = np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))
    L = s[-1] + np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1])
    if L <= 0:
        raise ValueError("Track có chu vi bằng 0 — kiểm tra lại file waypoint.")
    pts_c = np.vstack([pts, pts[0]])
    s_c = np.append(s, L)
    n = int(np.ceil(L / spacing))
    su = np.linspace(0.0, L - spacing, n)
    return np.column_stack([
        np.interp(su, s_c, pts_c[:, 0]),
        np.interp(su, s_c, pts_c[:, 1]),
    ])


def smooth_closed(pts: np.ndarray, window_m: float) -> np.ndarray:
    """Rolling average vòng kín, window tính theo mét."""
    n = len(pts)
    mean_sp = float(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])).mean())
    k = max(3, int(round(window_m / mean_sp)))
    if k % 2 == 0:
        k += 1
    if k >= n:
        k = n if n % 2 == 1 else n - 1
    kernel = np.ones(k) / k
    p3 = np.vstack([pts, pts, pts])
    x = np.convolve(p3[:, 0], kernel, "same")
    y = np.convolve(p3[:, 1], kernel, "same")
    return np.column_stack([x[n : 2 * n], y[n : 2 * n]])


# ---------------------------------------------------- curvature (đúng C++)
def curvature_5pt(pts: np.ndarray) -> np.ndarray:
    """Curvature 5-point span (±5 index) — GIỐNG HỆT mppi_xe.cpp."""
    w = len(pts)
    c = np.zeros(w)
    for i in range(w):
        p1 = pts[(i - 5 + w) % w]
        p2 = pts[i]
        p3 = pts[(i + 5) % w]
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = p3[0] - p2[0]
        dy2 = p3[1] - p2[1]
        l1 = np.hypot(dx1, dy1)
        l2 = np.hypot(dx2, dy2)
        l3 = np.hypot(p3[0] - p1[0], p3[1] - p1[1])
        c[i] = 4.0 * (dx1 * dy2 - dy1 * dx2) / (l1 * l2 * l3) if l1 * l2 * l3 > 1e-9 else 0.0
    return c


def curvature_arc(pts: np.ndarray, span_m: float) -> np.ndarray:
    """Curvature theo arc-length thật (span theo mét) — để kiểm chứng."""
    if span_m <= 0:
        return np.zeros(len(pts))
    n = max(1, int(round(span_m / 0.02)))
    w = len(pts)
    c = np.zeros(w)
    for i in range(w):
        p1 = pts[(i - n) % w]
        p2 = pts[i]
        p3 = pts[(i + n) % w]
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = p3[0] - p2[0]
        dy2 = p3[1] - p2[1]
        l1 = np.hypot(dx1, dy1)
        l2 = np.hypot(dx2, dy2)
        l3 = np.hypot(p3[0] - p1[0], p3[1] - p1[1])
        c[i] = 4.0 * (dx1 * dy2 - dy1 * dx2) / (l1 * l2 * l3) if l1 * l2 * l3 > 1e-9 else 0.0
    return c


def headings_5pt(pts: np.ndarray) -> np.ndarray:
    """Heading 5-point span — GIỐNG mppi_xe.cpp."""
    w = len(pts)
    h = np.zeros(w)
    for i in range(w):
        p1 = pts[(i - 5 + w) % w]
        p3 = pts[(i + 5) % w]
        h[i] = np.arctan2(p3[1] - p1[1], p3[0] - p1[0])
    return h


def deviation_from(orig: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Khoảng cách từ mỗi điểm mới tới raceline gốc (độ cắt cua)."""
    d = np.zeros(len(new))
    for i, p in enumerate(new):
        d[i] = np.hypot(orig[:, 0] - p[0], orig[:, 1] - p[1]).min()
    return d


# ------------------------------------------------------------- báo cáo
def stats_summary(pts: np.ndarray) -> dict:
    c5 = curvature_5pt(pts)
    c5a = np.abs(c5)
    ca = np.abs(curvature_arc(pts, 0.5))
    d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    d_c = np.append(d, np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1]))
    return {
        "n": len(pts),
        "length": float(d_c.sum()),
        "sp_min": float(d_c.min()),
        "sp_mean": float(d_c.mean()),
        "sp_max": float(d_c.max()),
        "curv_p50": float(np.percentile(c5a, 50)),
        "curv_p95": float(np.percentile(c5a, 95)),
        "curv_p99": float(np.percentile(c5a, 99)),
        "curv_max": float(c5a.max()),
        "pct_gt_1_0": float((c5a > 1.0).mean() * 100),
        "pct_gt_1_5": float((c5a > 1.5).mean() * 100),
        "arc_max": float(ca.max()),
    }


def print_summary(tag: str, st: dict) -> None:
    print(f"\n--- {tag} ---")
    print(f"  Điểm: {st['n']} | Chu vi: {st['length']:.1f} m")
    print(f"  Spacing (m): min={st['sp_min']:.3f} mean={st['sp_mean']:.3f} max={st['sp_max']:.3f}")
    print(f"  Curvature 5pt (như C++): p50={st['curv_p50']:.2f} p95={st['curv_p95']:.2f} "
          f"p99={st['curv_p99']:.2f} MAX={st['curv_max']:.2f} /m")
    print(f"  % điểm curv>1.0/m: {st['pct_gt_1_0']:.1f}% | curv>1.5/m: {st['pct_gt_1_5']:.1f}%")
    print(f"  Curvature arc-span 0.5m (thật): max={st['arc_max']:.2f} /m")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="Làm mượt + tái mẫu đều raceline F1TENTH (vòng kín)")
    ap.add_argument("--input", default="f1tenth_waypoint.csv",
                    help="File raceline đầu vào (mặc định: f1tenth_waypoint.csv)")
    ap.add_argument("--output", default=None,
                    help="File đầu ra (mặc định: <input>_smooth.csv)")
    ap.add_argument("--spacing", type=float, default=0.2,
                    help="Spacing resample cuối (m), mặc định 0.2")
    ap.add_argument("--window", type=float, default=0.0,
                    help="Window mượt (m). 0 = TỰ TÌM tăng dần tới khi đạt --max-curv (mặc định)")
    ap.add_argument("--max-curv", type=float, default=1.0,
                    help="Target curvature tối đa khi auto-window (1/m), mặc định 1.0")
    ap.add_argument("--win-max", type=float, default=4.0,
                    help="Window mượt tối đa khi auto (m), mặc định 4.0")
    ap.add_argument("--check-only", action="store_true",
                    help="Chỉ kiểm tra file, không tạo file mới")
    args = ap.parse_args()

    out_path = args.output or (args.input[:-4] + "_smooth.csv" if args.input.lower().endswith(".csv")
                               else args.input + "_smooth.csv")

    try:
        orig = load_waypoints(args.input)
    except ValueError as e:
        print(f"[LỖI] {e}", file=sys.stderr)
        return 1

    st_before = stats_summary(orig)
    print_summary("TRƯỚC", st_before)

    if args.check_only:
        print("\n[CHECK] Không tạo file (--check-only).")
        return 0

    # Nền tái mẫu mịn 0.15m trước khi mượt để window đồng nhất theo mét
    base = resample_uniform(orig, 0.15)

    if args.window > 0:
        windows = [args.window]
    else:
        windows = np.arange(0.3, args.win_max + 1e-9, 0.1)

    best = None
    for win in windows:
        smoothed = smooth_closed(base, float(win))
        final = resample_uniform(smoothed, args.spacing)
        curv_max = float(np.abs(curvature_5pt(final)).max())
        if best is None or curv_max < best[0]:
            best = (curv_max, win, final)
        if curv_max <= args.max_curv:
            break
    else:
        win_used, final = None, None
        # dùng kết quả mượt nhất tìm được
        _, win_used, final = best

    if args.window > 0:
        win_used = args.window
    else:
        win_used = best[1]

    st_after = stats_summary(final)
    print_summary("SAU", st_after)

    dev = deviation_from(orig, final)
    print(f"  Độ lệch so raceline gốc (m): p95={np.percentile(dev, 95):.2f} max={dev.max():.2f} "
          f"(cắt cua càng sâu nếu window lớn)")

    if st_after["curv_max"] <= args.max_curv:
        print(f"\n[OK] Đạt target: max curvature {st_after['curv_max']:.2f} ≤ {args.max_curv} /m "
              f"(window = {win_used:.1f} m)")
    else:
        print(f"\n[CẢNH BÁO] Không đạt target {args.max_curv} /m dù window tối đa {args.win_max} m. "
              f"Max còn {st_after['curv_max']:.2f} /m (bán kính ~{1/max(st_after['curv_max'],1e-9):.2f} m) — "
              f"xe F1TENTH chỉ quay được tối đa ~1.37 /m. Cần sửa lại track thật.")

    headings = headings_5pt(final)
    save_waypoints(out_path, final, headings)
    print(f"[ĐÃ GHI] {out_path} ({len(final)} điểm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
