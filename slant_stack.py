"""
Conventional benchmark: 4th-root slant stacking for vehicle detection and speed.

For each candidate slope (samples per channel) the channels are time-shifted
and stacked with an N-th root; moving vehicles appear as energy peaks in the
time-slope domain. Peaks above an adaptive (CDF-based) threshold inside the
allowed speed window are counted as vehicles, and each vehicle's speed comes
from the energy-weighted slope around its peak.

    slant_stack_from_npy()   one 180 s x 50-channel window (.npy)
    process_folder()         all .npy files in a folder -> vehicles.csv + summary.csv

Speeds are in mph throughout.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
from scipy.signal import convolve
from scipy.ndimage import median_filter

# ============================================================
# Unit helpers
# ============================================================
MPH2KMH = 1.60934
KMH2MPH = 1.0 / MPH2KMH


# ----------------------------
# Helpers
# ----------------------------
def gausswin(M: int, std: Optional[float] = None) -> np.ndarray:
    if M <= 1:
        return np.ones((1,), dtype=np.float32)
    if std is None:
        std = M / 6.0
    n = np.arange(M, dtype=np.float32) - (M - 1) / 2.0
    w = np.exp(-(n * n) / (2.0 * std * std))
    w /= (w.sum() + 1e-12)
    return w.astype(np.float32)


def corrcoef_channels(Data_TxC: np.ndarray) -> np.ndarray:
    return np.corrcoef(Data_TxC, rowvar=False)


def local_maxima_2d(A: np.ndarray, rmedge: int = 10) -> np.ndarray:
    """
    Find local maxima in a thresholded 2D array.
    Returns (time_index, slope_index) pairs.
    """
    T, S = A.shape
    if T <= 2 * rmedge + 2 or S <= 2 * rmedge + 2:
        return np.empty((0, 2), dtype=np.int32)

    core = A[rmedge:T - rmedge, rmedge:S - rmedge]
    xs, ys = np.nonzero(core > 0)
    xs = xs + rmedge
    ys = ys + rmedge

    peaks = []
    for x, y in zip(xs, ys):
        v = A[x, y]
        nb = A[x - 1:x + 2, y - 1:y + 2]
        if (v >= nb[0, 0] and v >  nb[0, 1] and v >= nb[0, 2] and
            v >  nb[1, 0] and v >  nb[1, 2] and
            v >= nb[2, 0] and v >  nb[2, 1] and v >= nb[2, 2]):
            peaks.append((x, y))
    return np.asarray(peaks, dtype=np.int32)


def remove_duplicate_peaks(
    time_idx: np.ndarray,
    slope_idx: np.ndarray,
    energy: np.ndarray,
    toltime_samples: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep only the max-energy peak within a time tolerance window.
    (time-only de-dup)
    """
    keep = np.ones_like(time_idx, dtype=bool)
    for i in range(len(time_idx)):
        if not keep[i]:
            continue
        close = np.where(np.abs(time_idx - time_idx[i]) < toltime_samples)[0]
        if len(close) <= 1:
            continue
        jmax = close[np.argmax(energy[close])]
        for j in close:
            if j != jmax:
                keep[j] = False
    return time_idx[keep], slope_idx[keep], energy[keep]


def shift_zeropad(x: np.ndarray, shift: int) -> np.ndarray:
    """
    Zero-padded shift (no wrap-around). Avoids np.roll artifacts.
    Positive shift => move forward (toward larger index).
    """
    y = np.zeros_like(x)
    if shift > 0:
        y[shift:] = x[:-shift]
    elif shift < 0:
        y[:shift] = x[-shift:]
    else:
        y[:] = x
    return y


def signed_speed_mph_from_slope(slope: float, dt: float, dx: float) -> float:
    v_kmh = 3.6 * dx / (abs(slope) * dt + 1e-12)
    v_mph = v_kmh * KMH2MPH
    return float(np.sign(slope) * v_mph)


# ----------------------------
# Config (ALL SPEEDS IN MPH)
# ----------------------------
@dataclass
class SlantStackConfig:
    channelspacing_m: float = 4.0

    # speeds are MPH (input + output)
    minspeed_mph: float = 6.0
    maxspeed_mph: float = 62.0

    use_qc: bool = True
    cc_topk: int = 2
    cc_thresh: float = 0.35
    max_bad_channels: int = 15

    Nroot: int = 4
    gauss_len: int = 125

    hist_bins: int = 200
    cdf_level: float = 0.68
    min_threshold: float = 100
    peak_gate: float = 0.98
    rmedge: int = 10
    peak_tol_sec: float = 3.0

    dslope: int = 1

    # suppress slopes near 0 by speed ceiling (MPH)
    zero_slope_v0_mph: float = 75.0

    # hard reject weak peaks: energy >= peak_min_factor * threshold
    peak_min_factor: float = 1.2

    # effective-speed settings (B mode)
    peak_slope_halfwin: int = 3     # +/- bins around the picked peak to compute effective slope
    use_zeropad_shift: bool = True  # True recommended; False to reproduce np.roll behavior


# ----------------------------
# Effective slope around each peak (B mode)
# ----------------------------
def effective_slope_at_peak(
    Stack_win: np.ndarray,
    slopes: np.ndarray,
    t_idx: int,
    s_idx: int,
    halfwin: int
) -> Tuple[float, float]:
    """
    Compute effective (energy-weighted) slope magnitude around the picked peak index.
    Returns:
      slope_eff_signed  (signed, samples/ch)
      slope_eff_abs     (abs, samples/ch)
    """
    Nslope = slopes.size
    j0 = max(0, s_idx - halfwin)
    j1 = min(Nslope, s_idx + halfwin + 1)

    local_E = Stack_win[t_idx, j0:j1].astype(np.float64)
    local_E = np.clip(local_E, 0.0, None)
    if local_E.sum() <= 0:
        # fallback to original picked slope
        s = float(slopes[s_idx])
        return s, abs(s)

    local_slopes = slopes[j0:j1].astype(np.float64)
    sign = np.sign(slopes[s_idx]) if slopes[s_idx] != 0 else 1.0
    local_abs = np.abs(local_slopes)

    slope_abs_eff = float(np.sum(local_abs * local_E) / (np.sum(local_E) + 1e-12))
    slope_eff_signed = float(sign * slope_abs_eff)
    return slope_eff_signed, slope_abs_eff


# ----------------------------
# Main core
# ----------------------------
def slant_stack_from_npy(
    npy_path: str,
    fs_hz: float,
    cfg: SlantStackConfig = SlantStackConfig(),
) -> Dict[str, Any]:
    dt = 1.0 / float(fs_hz)
    dx = float(cfg.channelspacing_m)

    # Convert mph inputs to km/h for slope window computations
    minspeed_kmh = float(cfg.minspeed_mph) * MPH2KMH
    maxspeed_kmh = float(cfg.maxspeed_mph) * MPH2KMH

    Z = np.load(npy_path)
    if Z.ndim != 2:
        raise ValueError("NPY must be a 2D array.")

    # Your export saves (C, T). Convert to (T, C).
    Data = Z.T.astype(np.float32, copy=False) if Z.shape[0] <= Z.shape[1] else Z.astype(np.float32, copy=False)
    T, C = Data.shape
    eps = 1e-12

    # ---------- QC ----------
    IndexBad = np.array([], dtype=np.int32)
    if cfg.use_qc and C >= (cfg.cc_topk + 2):
        CC = corrcoef_channels(Data)
        np.fill_diagonal(CC, 0.0)
        k = int(cfg.cc_topk)
        topk = np.partition(CC, -k, axis=0)[-k:, :]
        CCsum = np.mean(topk, axis=0)
        IndexBad = np.where(CCsum < cfg.cc_thresh)[0].astype(np.int32)

        if IndexBad.size > cfg.max_bad_channels:
            return {
                "ok": False,
                "reason": "too_many_bad_channels",
                "n_bad": int(IndexBad.size),
                "speed_mean_mph": np.nan,
                "speed_std_mph": -1.0 * float(IndexBad.size),
            }

        for idx in IndexBad:
            Data[:, idx] = np.median(Data[:, idx]).astype(np.float32)

    # ---------- Slope range ----------
    minspeed = minspeed_kmh
    maxspeed = maxspeed_kmh

    slope1 = int(-1 * round(3.6 * dx / (minspeed * dt)))
    slope2 = int(     round(3.6 * dx / (minspeed * dt)))
    slopes = np.arange(slope1, slope2 + 1, cfg.dslope, dtype=np.int32)
    Nslope = slopes.size

    nch1, nch2 = 0, C - 1
    nch0 = int(round(0.5 * (nch2 - nch1) + nch1))

    # ---------- Slant stacking ----------
    Stack = np.zeros((T, Nslope), dtype=np.float32)
    gwin = gausswin(cfg.gauss_len)
    denom = float(C - IndexBad.size / 2.0) if C > 0 else 1.0

    for ii, slope in enumerate(slopes):
        datasum = np.zeros(T, dtype=np.float32)
        for jj in range(nch2, nch1 - 1, -1):
            shift = int(-1 * slope * (jj - nch0))
            if cfg.use_zeropad_shift:
                tr = shift_zeropad(Data[:, jj], shift)
            else:
                tr = np.roll(Data[:, jj], shift)

            a = np.abs(tr)
            b = np.sign(tr) * np.power(a + eps, 1.0 / cfg.Nroot)
            datasum += b.astype(np.float32)

        datasum = datasum / denom
        a = np.abs(datasum)
        datasum = datasum * np.power(a + eps, (cfg.Nroot - 1))

        col = (100.0 * datasum).astype(np.float32)
        col = convolve(col, gwin, mode="same")
        Stack[:, ii] = col

    # ---------- Zero-slope suppression ----------
    v0_kmh = float(cfg.zero_slope_v0_mph) * MPH2KMH
    slope0 = 3.6 * dx / (v0_kmh * dt)
    slope0_bin = int(np.ceil(slope0))
    zero_band = np.where(np.abs(slopes) < slope0_bin)[0]
    if zero_band.size:
        Stack[:, zero_band] = 0.0

    # ---------- Threshold ----------
    StackMax = np.max(Stack, axis=1)
    StackMax = median_filter(StackMax, size=50, mode="nearest")

    counts, bin_edges = np.histogram(StackMax, bins=cfg.hist_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cdf = np.cumsum(counts).astype(np.float64)
    cdf /= (cdf[-1] + 1e-12)

    idx = np.where(cdf > cfg.cdf_level)[0]
    if idx.size == 0:
        thresholdLevel = cfg.min_threshold
        energy_thre = np.nan
    else:
        thresholdLevel = float(bin_centers[idx[0]])
        thresholdLevel = max(thresholdLevel, cfg.min_threshold)
        energy_thre = float(bin_centers[idx[0]])

    # ---------- Peak picking (restrict to speed window only) ----------
    sloperange1 = int(round(3.6 * dx / (minspeed * dt)))   # slow => big slope
    sloperange2 = int(round(3.6 * dx / (maxspeed * dt)))   # fast => small slope

    pos_range = np.where((slopes >= sloperange2) & (slopes <= sloperange1))[0]
    neg_range = np.where((slopes <= -sloperange2) & (slopes >= -sloperange1))[0]

    Stack_win = np.zeros_like(Stack, dtype=np.float32)
    if pos_range.size:
        Stack_win[:, pos_range] = Stack[:, pos_range]
    if neg_range.size:
        Stack_win[:, neg_range] = Stack[:, neg_range]

    StackPeaks = Stack_win * (Stack_win > cfg.peak_gate * thresholdLevel)
    peaks = local_maxima_2d(StackPeaks, rmedge=cfg.rmedge)

    if peaks.shape[0] < 1:
        tail = float(bin_centers[-1]) if bin_centers.size else thresholdLevel
        return {
            "ok": False,
            "reason": "no_peaks",
            "n_bad": int(IndexBad.size),
            "speed_mean_mph": np.nan,
            "speed_std_mph": -100.0 * tail,
            "thresholdLevel": float(thresholdLevel),
        }

    time_idx = peaks[:, 0]
    slope_idx = peaks[:, 1]

    # energy from Stack_win (consistent with peak picking)
    energy = Stack_win[time_idx, slope_idx]

    # time-only de-dup
    toltime_samples = cfg.peak_tol_sec / dt
    time_idx, slope_idx, energy = remove_duplicate_peaks(time_idx, slope_idx, energy, toltime_samples)

    Cslope_peak = slopes[slope_idx].astype(np.float32)
    time_sec = time_idx.astype(np.float32) * dt

    # HARD reject weak peaks
    good = energy >= (float(cfg.peak_min_factor) * thresholdLevel)
    time_idx = time_idx[good]
    slope_idx = slope_idx[good]
    Cslope_peak = Cslope_peak[good]
    time_sec = time_sec[good]
    Engy = energy[good]

    if Cslope_peak.size < 1:
        tail = float(bin_centers[-1]) if bin_centers.size else thresholdLevel
        return {
            "ok": False,
            "reason": "peaks_below_threshold_strict",
            "n_bad": int(IndexBad.size),
            "speed_mean_mph": np.nan,
            "speed_std_mph": -100.0 * tail,
            "thresholdLevel": float(thresholdLevel),
        }

    # strictly within speed window
    pos_mask = (Cslope_peak >= sloperange2) & (Cslope_peak <= sloperange1)
    neg_mask = (Cslope_peak <= -sloperange2) & (Cslope_peak >= -sloperange1)
    keep_mask = pos_mask | neg_mask

    time_idx = time_idx[keep_mask]
    slope_idx = slope_idx[keep_mask]
    Cslope_peak = Cslope_peak[keep_mask]
    time_sec = time_sec[keep_mask]
    Engy = Engy[keep_mask]

    if Cslope_peak.size < 1:
        return {
            "ok": False,
            "reason": "peaks_outside_speed_window",
            "n_bad": int(IndexBad.size),
            "speed_mean_mph": np.nan,
            "speed_std_mph": np.nan,
            "thresholdLevel": float(thresholdLevel),
        }

    # ---------- B mode: effective slope per peak ----------
    Cslope_eff = np.zeros_like(Cslope_peak, dtype=np.float32)
    Cslope_eff_abs = np.zeros_like(Cslope_peak, dtype=np.float32)

    for i in range(Cslope_peak.size):
        s_eff_signed, s_eff_abs = effective_slope_at_peak(
            Stack_win=Stack_win,
            slopes=slopes,
            t_idx=int(time_idx[i]),
            s_idx=int(slope_idx[i]),
            halfwin=int(cfg.peak_slope_halfwin),
        )
        Cslope_eff[i] = np.float32(s_eff_signed)
        Cslope_eff_abs[i] = np.float32(s_eff_abs)

    # per-peak effective speeds (mph)
    speed_each_eff_abs_mph = (3.6 * dx / (Cslope_eff_abs.astype(np.float64) * dt + 1e-12)) * KMH2MPH
    # optional: peak-point (original) speeds for reference
    speed_each_peak_abs_mph = (3.6 * dx / (np.abs(Cslope_peak.astype(np.float64)) * dt + 1e-12)) * KMH2MPH

    # summary mean speed: energy-weighted over peaks using effective slope (robust)
    w_peaks = Engy.astype(np.float64) / (Engy.sum() + 1e-12)
    # energy-weighted effective slope magnitude across peaks, then invert
    slope_eff_global = float(np.sum(Cslope_eff_abs.astype(np.float64) * w_peaks))
    speed_mean_mph = float((3.6 * dx / (slope_eff_global * dt + 1e-12)) * KMH2MPH)

    # std of per-peak effective speeds (optional dispersion)
    speed_std_mph = float(np.std(speed_each_eff_abs_mph)) if speed_each_eff_abs_mph.size else np.nan

    out: Dict[str, Any] = dict(
        ok=True,
        reason="ok",
        n_bad=int(IndexBad.size),
        thresholdLevel=float(thresholdLevel),
        energy_thre=float(energy_thre) if np.isfinite(energy_thre) else np.nan,

        # summary speeds (effective)
        speed_mean_mph=float(speed_mean_mph),
        speed_std_mph=float(speed_std_mph),

        # peak arrays
        peaks_time_sec=time_sec.astype(np.float32),

        # keep both slopes: peak vs effective
        peaks_slope_peak=Cslope_peak.astype(np.float32),
        peaks_slope_eff=Cslope_eff.astype(np.float32),
        peaks_energy=Engy.astype(np.float32),

        # per-peak speeds (mph): effective (primary) + peak (reference)
        peaks_speed_eff_abs_mph=speed_each_eff_abs_mph.astype(np.float32),
        peaks_speed_peak_abs_mph=speed_each_peak_abs_mph.astype(np.float32),

        fs_hz=float(fs_hz),
        dt=float(dt),
        dx=float(dx),

    )

    return out


# ============================================================
# Batch processing
# ============================================================
FNAME_RE = re.compile(r".*_(c\d+-\d+)_(\d{8})T(\d{6})_(\d{8})T(\d{6})\.npy$", re.IGNORECASE)

VEHICLE_FIELDS = ["file", "channel_span", "start_date", "start_time", "end_date", "end_time",
                  "vehicle_id_in_file", "peak_time_sec", "slope_peak", "slope_eff",
                  "speed_mph_abs", "speed_mph_signed", "speed_mph_abs_peak", "speed_mph_signed_peak",
                  "energy", "thresholdLevel", "n_bad_channels", "ok", "reason"]
SUMMARY_FIELDS = ["file", "channel_span", "start_date", "start_time", "end_date", "end_time", "ok", "reason",
                  "n_bad_channels", "thresholdLevel", "energy_thre", "speed_mean_mph", "speed_std_mph", "n_peaks"]


def parse_meta_from_filename(path: str):
    """``(channel_span, YYYY-MM-DD, HH:MM:SS, YYYY-MM-DD, HH:MM:SS)`` or Nones."""
    m = FNAME_RE.match(os.path.basename(path))
    if not m:
        return None, None, None, None, None
    span, sd, st, ed, et = m.groups()
    d = lambda s: f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    t = lambda s: f"{s[0:2]}:{s[2:4]}:{s[4:6]}"
    return span, d(sd), t(st), d(ed), t(et)


def _num(v, cast=float):
    return cast(v) if v is not None and np.isfinite(v) else ""


def process_folder(npy_dir: str, out_dir: str, fs_hz: float = 100.0,
                   cfg: SlantStackConfig = SlantStackConfig(), max_files: Optional[int] = None) -> None:
    """Run slant stacking on every .npy in ``npy_dir``; one row per vehicle / per file."""
    files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))[:max_files]
    if not files:
        raise FileNotFoundError(f"No .npy found in: {npy_dir}")
    os.makedirs(out_dir, exist_ok=True)
    out_veh, out_sum = os.path.join(out_dir, "vehicles.csv"), os.path.join(out_dir, "summary.csv")
    t0 = time.perf_counter()

    with open(out_veh, "w", newline="", encoding="utf-8") as fv, open(out_sum, "w", newline="", encoding="utf-8") as fs:
        wv, ws = csv.DictWriter(fv, VEHICLE_FIELDS), csv.DictWriter(fs, SUMMARY_FIELDS)
        wv.writeheader()
        ws.writeheader()
        for k, path in enumerate(files, start=1):
            span, sd, st, ed, et = parse_meta_from_filename(path)
            meta = dict(file=os.path.basename(path), channel_span=span, start_date=sd, start_time=st,
                        end_date=ed, end_time=et)
            try:
                res = slant_stack_from_npy(path, fs_hz=fs_hz, cfg=cfg)
            except Exception as e:
                print(f"[ERROR] {os.path.basename(path)} -> {type(e).__name__}: {e}")
                res = {"ok": False, "reason": f"exception: {type(e).__name__}: {e}"}

            ok = bool(res.get("ok", False))
            thr = res.get("thresholdLevel", np.nan)
            times = res.get("peaks_time_sec", np.array([], dtype=np.float32))
            ws.writerow({**meta, "ok": ok, "reason": res.get("reason", ""),
                         "n_bad_channels": _num(res.get("n_bad", np.nan), int), "thresholdLevel": _num(thr),
                         "energy_thre": _num(res.get("energy_thre", np.nan)),
                         "speed_mean_mph": _num(res.get("speed_mean_mph", np.nan)),
                         "speed_std_mph": _num(res.get("speed_std_mph", np.nan)),
                         "n_peaks": int(times.size) if ok else 0})
            if not ok or times.size == 0:
                continue

            dt, dx = res["dt"], res["dx"]
            for i in range(times.size):
                s_peak, s_eff = float(res["peaks_slope_peak"][i]), float(res["peaks_slope_eff"][i])
                v_eff = signed_speed_mph_from_slope(s_eff, dt=dt, dx=dx)
                v_peak = signed_speed_mph_from_slope(s_peak, dt=dt, dx=dx)
                wv.writerow({**meta, "vehicle_id_in_file": i + 1, "peak_time_sec": float(times[i]),
                             "slope_peak": s_peak, "slope_eff": s_eff,
                             "speed_mph_abs": abs(v_eff), "speed_mph_signed": v_eff,
                             "speed_mph_abs_peak": abs(v_peak), "speed_mph_signed_peak": v_peak,
                             "energy": float(res["peaks_energy"][i]), "thresholdLevel": _num(thr),
                             "n_bad_channels": _num(res.get("n_bad", np.nan), int), "ok": True, "reason": "ok"})
            if k % 50 == 0:
                print(f"[{k}/{len(files)}] processed...")

    print(f"Saved {out_veh} and {out_sum} in {time.perf_counter() - t0:.1f} s")


# Settings used in the paper
PAPER_CONFIG = SlantStackConfig(
    channelspacing_m=4.0, minspeed_mph=10.0, maxspeed_mph=70.0, zero_slope_v0_mph=75.0,
    use_qc=True, cc_thresh=0.35, peak_gate=0.99, peak_min_factor=1.2, peak_tol_sec=3.0,
    peak_slope_halfwin=3, use_zeropad_shift=True,
)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npy", required=True, help="folder of .npy windows")
    p.add_argument("--out", required=True)
    p.add_argument("--fs", type=float, default=100.0)
    p.add_argument("--max-files", type=int, default=None)
    a = p.parse_args()
    process_folder(a.npy, a.out, fs_hz=a.fs, cfg=PAPER_CONFIG, max_files=a.max_files)
