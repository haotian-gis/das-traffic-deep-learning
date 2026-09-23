"""
Vehicle detection and apparent-speed estimation on DAS spatiotemporal images.

Each detected bounding box contains one bright trajectory. Its slope is found
with PCA on the brightest pixels; the slope converts to a signed apparent speed
(time on x, channels on y). Poor fits fall back to multi-threshold PCA and a
2-cluster split along y, and the most plausible candidate is kept.

    run_detector()    YOLO inference + speed per detection         -> <out>/speeds.csv
    run_benchmark()   same speed estimation on manual YOLO labels  -> <out>/speeds.csv

Both write identical columns, so detector and manual results compare directly.
"""
import argparse
import csv
import os
import re
from pathlib import Path

import cv2
import numpy as np

import config

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

BRIGHT_Q = 80          # percentile threshold for trajectory pixels
MIN_PTS = 400          # minimum trajectory pixels for a fit
DIR_SIGN = -1          # flips the sign convention of travel direction
MAX_ABS_MPH = 75.0     # plausibility limit

CSV_COLUMNS = ["image", "channel_span", "start_date", "start_time", "end_date", "end_time",
               "vehicle_id_in_file", "det_idx", "cls", "bbox", "method", "score", "is_fallback",
               "slope_dy_dx", "v_mps_signed", "v_kmh_signed", "v_mph_signed", "n_pts", "ratio", "abs_dx"]

# e.g. f015_c2675-2724_20231101T000154_20231101T000454[...]
FILENAME_RE = re.compile(r"^(?P<prefix>.+?)_(?P<span>[cC]\d+-\d+)_(?P<start>\d{8}T\d{6})_(?P<end>\d{8}T\d{6}).*$")


# =============================================================================
# Helpers
# =============================================================================
def _split_dt(s):
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}", f"{s[9:11]}:{s[11:13]}:{s[13:15]}"


def parse_image_stem(stem):
    """``(channel_span, start_date, start_time, end_date, end_time)`` from an image name."""
    m = FILENAME_RE.match(stem)
    if not m:
        return "", "", "", "", ""
    return (m.group("span"), *_split_dt(m.group("start")), *_split_dt(m.group("end")))


def list_images(img_dir):
    return sorted(p for ext in IMG_EXTS for p in Path(img_dir).glob(f"*{ext}"))


def speed_constants(width, height, window_s=config.WINDOW_SEC,
                    n_ch=config.CHANNELS_PER_IMAGE, dx_m=config.CHANNEL_SPACING_M):
    """Pixel slope (dy/dx) -> speed factors in m/s, km/h and mph."""
    k_mps = ((n_ch * dx_m) / height) / (window_s / width)
    k_kmh = k_mps * 3.6
    return k_mps, k_kmh, k_kmh / 1.60934


def yolo_label_to_boxes(txt_path, width, height):
    """YOLO ``cls cx cy w h`` (normalised) -> list of ``(cls, (x1, y1, x2, y2))`` in pixels."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    for line in open(txt_path, encoding="utf-8"):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        x1, y1 = max(0, int((cx - bw / 2) * width)), max(0, int((cy - bh / 2) * height))
        x2, y2 = min(width - 1, int((cx + bw / 2) * width)), min(height - 1, int((cy + bh / 2) * height))
        if x2 > x1 + 5 and y2 > y1 + 5:
            boxes.append((int(float(p[0])), (x1, y1, x2, y2)))
    return boxes


# =============================================================================
# Trajectory fitting
# =============================================================================
def _gray_roi(img, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)


def _pca_line(pts, x1, y1):
    """Principal direction of a point cloud -> (slope, line endpoints, singular values, dx) or None."""
    mean = pts.mean(axis=0)
    pts0 = pts - mean
    _, svals, vt = np.linalg.svd(pts0, full_matrices=False)
    dx, dy = float(vt[0][0]), float(vt[0][1])
    if abs(dx) < 1e-6:
        return None
    t = pts0 @ vt[0]
    p1, p2 = mean + t.min() * vt[0], mean + t.max() * vt[0]
    line = (int(p1[0] + x1), int(p1[1] + y1), int(p2[0] + x1), int(p2[1] + y1))
    return dy / dx, line, svals, dx


def fit_trajectory_pca(img, box, q=BRIGHT_Q):
    """Signed pixel slope of the trajectory in ``box`` -> ``(slope, line, stats)`` or None."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 + 5 or y2 <= y1 + 5:
        return None
    gray = _gray_roi(img, (x1, y1, x2, y2))
    if gray is None:
        return None
    ys, xs = np.where(gray >= np.percentile(gray, q))
    if xs.size < MIN_PTS:
        return None
    res = _pca_line(np.column_stack([xs, ys]).astype(np.float32), x1, y1)
    if res is None:
        return None
    slope, line, svals, dx = res
    return slope, line, {"n_pts": int(xs.size), "abs_dx": abs(dx), "ratio": float(svals[0] / (svals[1] + 1e-6)), "q": q}


def fit_score(stats, slope, k_mph):
    """Higher = more elongated, less vertical and more plausible speed."""
    mph = abs(slope) * k_mph
    return (stats["ratio"] * stats["abs_dx"]) / (1.0 + (mph / MAX_ABS_MPH) ** 2)


def is_abnormal(v_mph, stats):
    return (not np.isfinite(v_mph) or abs(v_mph) > MAX_ABS_MPH
            or stats.get("abs_dx", 1.0) < 0.18 or stats.get("ratio", 999.0) < 1.8)


def fit_trajectory_fallback(img, box, k_mph):
    """Best of multi-threshold PCA and 2-cluster (along y) PCA -> ``(slope, line, stats, method, score)`` or None."""
    candidates = []
    for q in (BRIGHT_Q, 88, 92, 75):
        r = fit_trajectory_pca(img, box, q=q)
        if r is not None:
            candidates.append((fit_score(r[2], r[0], k_mph), f"pca_q{q}", *r))

    x1, y1 = int(box[0]), int(box[1])
    gray = _gray_roi(img, box)
    if gray is not None:
        ys, xs = np.where(gray >= np.percentile(gray, BRIGHT_Q))
        if xs.size >= MIN_PTS * 2:
            pts = np.column_stack([xs, ys]).astype(np.float32)
            yv = pts[:, 1].reshape(-1, 1)
            c1, c2 = float(np.percentile(yv, 25)), float(np.percentile(yv, 75))
            lab = None
            for _ in range(15):                          # 1-D k-means with k = 2
                lab = (np.abs(yv - c2) < np.abs(yv - c1)).astype(np.int32).ravel()
                if min((lab == 0).sum(), (lab == 1).sum()) < MIN_PTS // 2:
                    break
                n1, n2 = float(yv[lab == 0].mean()), float(yv[lab == 1].mean())
                if abs(n1 - c1) < 1e-3 and abs(n2 - c2) < 1e-3:
                    break
                c1, c2 = n1, n2
            if lab is not None:
                for k in (0, 1):
                    pk = pts[lab == k]
                    if len(pk) < MIN_PTS:
                        continue
                    res = _pca_line(pk, x1, y1)
                    if res is None:
                        continue
                    slope, line, svals, dx = res
                    stats = {"n_pts": int(len(pk)), "abs_dx": abs(dx),
                             "ratio": float(svals[0] / (svals[1] + 1e-6)), "q": BRIGHT_Q, "cluster": k}
                    candidates.append((fit_score(stats, slope, k_mph), f"cluster_y{k}", slope, line, stats))

    if not candidates:
        return None
    score, method, slope, line, stats = max(candidates, key=lambda c: c[0])
    return slope, line, stats, method, float(score)


def estimate_speed(img, box, consts):
    """Full estimation for one box. Returns a dict with slope, speeds, method, score and stats."""
    k_mps, k_kmh, k_mph = consts
    out = dict(slope=np.nan, v_mps=np.nan, v_kmh=np.nan, v_mph=np.nan, method=f"pca_q{BRIGHT_Q}",
               score=np.nan, fallback=False, stats={}, line=tuple(int(v) for v in box))
    base = fit_trajectory_pca(img, box)
    if base is None:
        return out

    slope, line, stats = base
    v_mph = DIR_SIGN * slope * k_mph
    score = fit_score(stats, slope, k_mph)
    method, fallback = out["method"], False

    if is_abnormal(v_mph, stats):
        better = fit_trajectory_fallback(img, box, k_mph)
        if better is not None:
            s2, l2, st2, m2, sc2 = better
            v2 = DIR_SIGN * s2 * k_mph
            if np.isfinite(v2) and abs(v2) <= MAX_ABS_MPH and sc2 > score:
                slope, line, stats, method, score, fallback = s2, l2, st2, m2, sc2, True

    out.update(line=line, stats=stats, method=method, score=score, fallback=fallback)
    v_mph = DIR_SIGN * slope * k_mph
    if np.isfinite(v_mph) and abs(v_mph) <= MAX_ABS_MPH * 1.5:   # otherwise keep NaN
        out.update(slope=slope, v_mps=DIR_SIGN * slope * k_mps, v_kmh=DIR_SIGN * slope * k_kmh, v_mph=v_mph)
    return out


def _row(stem, j, cls, box, est):
    st = est["stats"]
    return [stem, *parse_image_stem(stem), j, j, int(cls), tuple(int(v) for v in box), est["method"],
            f"{est['score']:.4f}" if np.isfinite(est["score"]) else "nan", int(est["fallback"]),
            est["slope"], est["v_mps"], est["v_kmh"], est["v_mph"], st.get("n_pts", ""),
            f"{st.get('ratio', 0):.3f}" if st else "", f"{st.get('abs_dx', 0):.3f}" if st else ""]


def draw(vis, box, est):
    """Overlay box (yellow), fitted line (green) and speed (red)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.line(vis, est["line"][:2], est["line"][2:], (0, 255, 0), 2)
    text = (f"{est['v_mph']:+.1f} mph" if np.isfinite(est["v_mph"]) else "NaN") + ("*" if est["fallback"] else "")
    cv2.putText(vis, text, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)


# =============================================================================
# Runs
# =============================================================================
def run_benchmark(img_dir, label_dir, out_dir, save_vis=True):
    """Speed estimation on manually labelled boxes (benchmark)."""
    out_dir = config.ensure_dir(out_dir)
    vis_dir = config.ensure_dir(out_dir / "labels_vis") if save_vis else None
    imgs = list_images(img_dir)
    if not imgs:
        raise FileNotFoundError(f"No images in {img_dir}")
    with open(out_dir / "speeds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for p in imgs:
            img = cv2.imread(str(p))
            if img is None:
                continue
            H, W = img.shape[:2]
            consts = speed_constants(W, H)
            vis = img.copy()
            for j, (cls, box) in enumerate(yolo_label_to_boxes(Path(label_dir) / f"{p.stem}.txt", W, H)):
                est = estimate_speed(img, box, consts)
                w.writerow(_row(p.stem, j, cls, box, est))
                draw(vis, box, est)
            if save_vis:
                cv2.imwrite(str(vis_dir / f"{p.stem}_vis.png"), vis)
    print(f"Saved {out_dir / 'speeds.csv'}")


def run_detector(img_dir, out_dir, weights=config.BEST_WEIGHTS, conf=0.25, iou=0.65, batch=2,
                 half=True, max_det=300, save_vis=False, export_labels=False):
    """YOLO inference + speed estimation. Halves the batch on CUDA out-of-memory."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from ultralytics import YOLO

    out_dir = config.ensure_dir(out_dir)
    vis_dir = config.ensure_dir(out_dir / "vis") if save_vis else None
    lbl_dir = config.ensure_dir(out_dir / "labels_predicted") if export_labels else None
    imgs = list_images(img_dir)
    if not imgs:
        raise FileNotFoundError(f"No images in {img_dir}")
    model = YOLO(str(weights))
    torch.set_grad_enabled(False)

    with open(out_dir / "speeds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        i = 0
        while i < len(imgs):
            chunk = imgs[i:i + batch]
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                results = model.predict(source=[str(p) for p in chunk], imgsz=config.IMG_SIZE, conf=conf, iou=iou,
                                        device=config.DEVICE, half=half, batch=batch, stream=True,
                                        verbose=False, max_det=max_det, save=False)
                n_det = 0
                for r in results:
                    stem = Path(r.path).stem
                    img = cv2.imread(r.path)
                    if img is None:
                        continue
                    consts = speed_constants(img.shape[1], img.shape[0])
                    vis = img.copy()
                    if r.boxes is not None and len(r.boxes):
                        cls = r.boxes.cls.cpu().numpy().astype(int)
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        if export_labels:
                            xywhn = r.boxes.xywhn.cpu().numpy()
                            (lbl_dir / f"{stem}.txt").write_text(
                                "".join(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n" for c, (x, y, bw, bh) in zip(cls, xywhn)))
                        for j in range(len(cls)):
                            est = estimate_speed(img, xyxy[j], consts)
                            w.writerow(_row(stem, j, cls[j], xyxy[j], est))
                            if save_vis:
                                draw(vis, xyxy[j], est)
                            n_det += 1
                    if save_vis:
                        cv2.imwrite(str(vis_dir / f"{stem}_vis.png"), vis)
                i += len(chunk)
                print(f"[OK] {i}/{len(imgs)} | batch={batch} | dets={n_det}")
            except torch.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                print(f"[OOM] retry with batch={batch}")
    print(f"Saved {out_dir / 'speeds.csv'}")


def run_days(days=None, **kwargs):
    """Run the detector for each day folder ``IMAGES_DIR/<YYYYMMDD>`` -> ``RESULTS_DIR/<YYYYMMDD>``."""
    days = days or sorted(p.name for p in Path(config.IMAGES_DIR).iterdir() if p.is_dir())
    for d in days:
        run_detector(Path(config.IMAGES_DIR) / d, Path(config.RESULTS_DIR) / d, **kwargs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--images", required=True)
    b.add_argument("--labels", required=True)
    b.add_argument("--out", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--images", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--weights", default=str(config.BEST_WEIGHTS))
    d.add_argument("--save-vis", action="store_true")
    days = sub.add_parser("days")
    days.add_argument("dates", nargs="*", help="YYYYMMDD folder names (default: all)")
    a = p.parse_args()
    if a.mode == "benchmark":
        run_benchmark(a.images, a.labels, Path(a.out))
    elif a.mode == "detect":
        run_detector(a.images, Path(a.out), weights=a.weights, save_vis=a.save_vis)
    else:
        run_days(a.dates or None)
