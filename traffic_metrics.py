"""
Traffic-state indicators from vehicle-level detections.

Validation (single day with manual labels)
    compare_methods()          3-min vehicle counts / mean speeds: benchmark vs slant stacking vs YOLO
    count_metrics()            r, R², Spearman ρ, MAE, RMSE, MAPE, bias of counts vs benchmark
    agreement_30min()          30-min volume & speed agreement, slant stacking vs YOLO (linear fit, R²)

Network level (Oct-Nov 2023)
    network_timeseries()       volume (sum) and mean |speed| at 15/30-min resolution
    event_diurnal_profile()    congestion intensity (volume / speed) of an event vs matched baseline

Segment level (50-channel roadway segments)
    aggregate_segments()       per segment and 15-min bin: directional flows and mean speeds
    segment_event_change()     mean % change in congestion intensity, event vs baseline, per segment
    segment_diurnal_profile()  diurnal profile of one segment, event vs baseline mean ± SD

``python traffic_metrics.py`` writes all tables to ``config.OUTPUT_DIR``.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config

MAX_SPEED_MPH = 70.0     # speeds above are ignored in the 3-min comparison
EPS_SPEED = 0.1          # avoids division by zero in network-level congestion intensity


# =============================================================================
# Loading
# =============================================================================
def build_datetime(df):
    """Window start time from ``datetime`` or ``start_date`` + ``start_time``."""
    cols = df.columns.str.lower()
    if "datetime" in cols:
        return pd.to_datetime(df[df.columns[cols == "datetime"][0]], errors="coerce")
    if "start_date" in cols and "start_time" in cols:
        d, t = df.columns[cols == "start_date"][0], df.columns[cols == "start_time"][0]
        return pd.to_datetime(df[d].astype(str) + " " + df[t].astype(str), errors="coerce")
    raise KeyError("Need 'datetime' or 'start_date' + 'start_time'.")


def load_detections(*csv_paths):
    """Concatenate detection CSVs (speed_estimation / slant_stack output) with a ``datetime`` column."""
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    df["datetime"] = build_datetime(df)
    return df.dropna(subset=["datetime"]).copy()


# =============================================================================
# Validation against manual labels and slant stacking
# =============================================================================
def compare_methods(benchmark, slant, yolo, max_speed=MAX_SPEED_MPH, rolling="15min"):
    """
    Per 3-min window: vehicle counts of the three methods and mean |speed|
    of slant stacking and YOLO, plus ``*_roll`` rolling means.
    """
    counts = pd.concat([benchmark.groupby("datetime").size().rename("benchmark"),
                        slant.groupby("datetime").size().rename("slope"),
                        yolo.groupby("datetime").size().rename("yolo")], axis=1).dropna().sort_index()

    v_yolo = yolo["v_mph_signed"].abs()
    v_slant = slant["speed_mph_abs"]
    speeds = pd.concat([
        v_slant.where(v_slant <= max_speed).groupby(slant["datetime"]).mean().rename("slope"),
        v_yolo.where(v_yolo <= max_speed).groupby(yolo["datetime"]).mean().rename("yolo"),
    ], axis=1).dropna().sort_index()

    for df in (counts, speeds):
        for c in list(df.columns):
            df[f"{c}_roll"] = df[c].rolling(rolling, min_periods=1).mean()
    return counts.rename_axis("datetime").reset_index(), speeds.rename_axis("datetime").reset_index()


def count_metrics(df, pred_col, gt_col="benchmark"):
    d = df[[gt_col, pred_col]].dropna()
    gt, pr = d[gt_col].to_numpy(float), d[pred_col].to_numpy(float)
    err = pr - gt
    r = np.corrcoef(gt, pr)[0, 1] if len(gt) > 1 else np.nan
    rho, p = spearmanr(gt, pr) if len(gt) > 1 else (np.nan, np.nan)
    nz = gt != 0
    return {"method": pred_col, "N": len(gt), "Pearson_r": r, "R2": r ** 2, "Spearman_rho": rho, "Spearman_p": p,
            "MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(np.mean(err ** 2)),
            "MAPE_%": np.mean(np.abs(err[nz] / gt[nz])) * 100 if nz.any() else np.nan,
            "MAPE_N": int(nz.sum()), "Bias": np.mean(err)}


def speed_correlation(speeds):
    x, y = speeds["slope"].to_numpy(float), speeds["yolo"].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    r = np.corrcoef(x[m], y[m])[0, 1]
    rho, p = spearmanr(x[m], y[m])
    return {"Pearson_r": r, "R2": r ** 2, "Spearman_rho": rho, "Spearman_p": p}


def linear_fit(x, y):
    """Least-squares ``y = a x + b`` and R² of the fit."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    a, b = np.polyfit(x, y, 1)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {"slope": a, "intercept": b, "R2": 1 - np.sum((y - (a * x + b)) ** 2) / ss_tot if ss_tot > 0 else np.nan,
            "N": int(m.sum())}


def agreement_30min(slant, yolo, rule="30min"):
    """30-min vehicle volume and mean |speed|, slant stacking vs YOLO, with linear fits."""
    counts = pd.concat([slant.groupby("datetime").size().rename("slope"),
                        yolo.groupby("datetime").size().rename("yolo")], axis=1).dropna()
    counts.index = pd.to_datetime(counts.index).floor("3min")
    table = counts.resample(rule, label="left", closed="left").sum().dropna()

    def mean_speed(df, col, name):
        s = pd.to_numeric(df[col], errors="coerce").abs()
        return (pd.DataFrame({"t": df["datetime"].dt.floor("3min"), "v": s}).dropna()
                .set_index("t")["v"].resample(rule, label="left", closed="left").mean().rename(name))

    table = table.join(mean_speed(slant, "speed_mph_abs", "slope_speed"), how="left") \
                 .join(mean_speed(yolo, "v_mph_signed", "yolo_speed"), how="left")
    fits = pd.DataFrame([{"quantity": "volume", **linear_fit(table["slope"].to_numpy(float), table["yolo"].to_numpy(float))},
                         {"quantity": "speed", **linear_fit(table["slope_speed"].to_numpy(float),
                                                            table["yolo_speed"].to_numpy(float))}])
    return table.rename_axis("datetime").reset_index(), fits


# =============================================================================
# Network-level time series and events
# =============================================================================
def network_timeseries(yolo, rule="15min"):
    """Vehicle volume (sum) and mean |speed| (mph) over all channels per ``rule`` bin."""
    counts = yolo.groupby("datetime").size().rename("yolo").to_frame()
    counts.index = pd.to_datetime(counts.index).floor("3min")
    counts = counts.groupby(counts.index).sum()
    spd = pd.DataFrame({"datetime": yolo["datetime"].dt.floor("3min"),
                        "abs_speed": pd.to_numeric(yolo["v_mph_signed"], errors="coerce").abs()}).dropna()

    out = counts.resample(rule, label="left", closed="left").sum() \
                .join(spd.set_index("datetime")["abs_speed"].resample(rule, label="left", closed="left")
                      .mean().rename("yolo_speed"), how="left")
    out = out.rename_axis("datetime").reset_index()
    out["weekday"] = out["datetime"].dt.day_name()
    out["congestion_intensity"] = out["yolo"] / (out["yolo_speed"] + EPS_SPEED)
    out["home_gameday"] = out["datetime"].dt.normalize().isin(pd.to_datetime(config.HOME_GAMEDAYS))
    return out


def event_diurnal_profile(ts, event_days, baseline_days, value="congestion_intensity"):
    """
    Mean ± SD of ``value`` by time of day over the baseline days, the mean over
    the event days, and their difference. Also returns the 24-h mean % change
    (event mean vs the average of the baseline days' daily means).
    """
    df = ts.dropna(subset=["datetime"]).copy()
    df["date"] = df["datetime"].dt.normalize()
    df["tod"] = df["datetime"].dt.strftime("%H:%M")
    ev, base = pd.to_datetime(event_days).normalize(), pd.to_datetime(baseline_days).normalize()
    df_ev, df_base = df[df["date"].isin(ev)], df[df["date"].isin(base)]

    prof = df_base.groupby("tod")[value].agg(baseline_mean="mean", baseline_std="std").reset_index() \
        .merge(df_ev.groupby("tod")[value].mean().rename("event").reset_index(), on="tod", how="outer")
    t = pd.to_datetime(prof["tod"], format="%H:%M")
    prof["tod_min"] = t.dt.hour * 60 + t.dt.minute
    prof = prof.sort_values("tod_min").reset_index(drop=True)
    prof["baseline_low"] = prof["baseline_mean"] - prof["baseline_std"]
    prof["baseline_high"] = prof["baseline_mean"] + prof["baseline_std"]
    prof["abs_change"] = prof["event"] - prof["baseline_mean"]
    prof["pct_change"] = prof["abs_change"] / prof["baseline_mean"] * 100

    base_24h = df_base.groupby("date")[value].mean().mean()
    pct_24h = (df_ev[value].mean() - base_24h) / (base_24h + 1e-12) * 100
    return prof, pct_24h


# =============================================================================
# Segment level
# =============================================================================
def aggregate_segments(speeds_csv, freq="15min", seg_start=config.SEG_START_CH, seg_size=config.SEG_SIZE,
                       qc_abs_dx_max=None, qc_npts_min=None):
    """
    For each (seg, channel_span, time_bin): ``flow_pos`` / ``flow_neg`` (by sign of
    ``v_kmh_signed``), ``flow_total`` (all detections, incl. NaN speed) and mean |speed|
    (km/h) per direction and overall.
    """
    df = pd.read_csv(speeds_csv)
    df["start_datetime"] = pd.to_datetime(df["start_date"] + " " + df["start_time"], errors="coerce")
    df["speed_signed"] = df["v_kmh_signed"]
    df["speed_abs"] = df["speed_signed"].abs()

    span = df["channel_span"].str.extract(r"c(?P<ch0>\d+)-(?P<ch1>\d+)")
    if span.isna().any(axis=None):
        raise ValueError("Unparseable channel_span: " + ", ".join(df.loc[span.isna().any(axis=1), "channel_span"].head(10)))
    df["ch_start"], df["ch_end"] = span["ch0"].astype(int), span["ch1"].astype(int)
    df["seg"] = ((df["ch_start"] - seg_start) // seg_size).astype(int)
    if not (df["ch_end"] - df["ch_start"] + 1).eq(seg_size).all():
        print(f"Warning: some channel spans are not {seg_size} channels wide.")

    if qc_abs_dx_max is not None:
        df = df[df["abs_dx"].notna() & (df["abs_dx"] <= qc_abs_dx_max)]
    if qc_npts_min is not None:
        df = df[df["n_pts"].notna() & (df["n_pts"] >= qc_npts_min)]
    df = df.assign(time_bin=df["start_datetime"].dt.floor(freq))

    valid = df["speed_signed"].notna()
    df = df.assign(pos=valid & (df["speed_signed"] > 0), neg=valid & (df["speed_signed"] < 0), valid=valid)
    g = df.groupby(["seg", "channel_span", "time_bin"], dropna=False)
    out = pd.DataFrame({
        "flow_pos": g["pos"].sum().astype(int),
        "flow_neg": g["neg"].sum().astype(int),
        "flow_total": g.size().astype(int),
        "speed_pos_mean": df["speed_abs"].where(df["pos"]).groupby([df["seg"], df["channel_span"], df["time_bin"]], dropna=False).mean(),
        "speed_neg_mean": df["speed_abs"].where(df["neg"]).groupby([df["seg"], df["channel_span"], df["time_bin"]], dropna=False).mean(),
        "speed_total_mean": df["speed_abs"].where(df["valid"]).groupby([df["seg"], df["channel_span"], df["time_bin"]], dropna=False).mean(),
    }).reset_index().sort_values(["seg", "channel_span", "time_bin"])
    return out


def aggregate_all_days(results_dir=config.RESULTS_DIR, **kwargs):
    """Run :func:`aggregate_segments` for every ``<day>/speeds.csv`` and save ``aggregated_speeds.csv``."""
    for csv in sorted(Path(results_dir).glob("*/speeds.csv")):
        out = csv.parent / "aggregated_speeds.csv"
        aggregate_segments(csv, **kwargs).to_csv(out, index=False)
        print(f"Saved {out}")


def load_segment_tables(days, results_dir=config.RESULTS_DIR):
    """Aggregated segment tables of the given days with congestion intensity and time of day."""
    files = [Path(results_dir) / pd.Timestamp(d).strftime("%Y%m%d") / "aggregated_speeds.csv" for d in days]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError("Missing:\n" + "\n".join(missing))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["congestion_intensity"] = (df["flow_total"] / df["speed_total_mean"]).replace([np.inf, -np.inf], np.nan)
    df["time_bin"] = pd.to_datetime(df["time_bin"], errors="coerce")
    df = df.dropna(subset=["time_bin", "seg", "congestion_intensity"])
    df["date"] = df["time_bin"].dt.normalize()
    df["time_of_day"] = df["time_bin"].dt.strftime("%H:%M")
    return df


def segment_event_change(df, event_days, baseline_days):
    """
    % change in congestion intensity per segment and time of day
    (event-day mean vs baseline-day mean), then averaged over the day.
    """
    ev, base = pd.to_datetime(event_days).normalize(), pd.to_datetime(baseline_days).normalize()
    key = ["seg", "time_of_day"]
    m = df[df["date"].isin(ev)].groupby(key)["congestion_intensity"].mean().rename("event").reset_index() \
        .merge(df[df["date"].isin(base)].groupby(key)["congestion_intensity"].mean().rename("baseline").reset_index(),
               on=key, how="inner")
    m["pct_change"] = ((m["event"] - m["baseline"]) / m["baseline"] * 100).replace([np.inf, -np.inf], np.nan)
    m = m.dropna(subset=["pct_change"])
    return m.groupby("seg")["pct_change"].mean().rename("pct_change_mean").reset_index(), m


def segment_diurnal_profile(df, seg, event_days, baseline_days):
    """Congestion intensity of one segment by time of day: event mean vs baseline mean ± SD."""
    ev, base = pd.to_datetime(event_days).normalize(), pd.to_datetime(baseline_days).normalize()
    s = df[df["seg"] == seg]
    prof = s[s["date"].isin(ev)].groupby("time_of_day")["congestion_intensity"].mean().rename("event").reset_index() \
        .merge(s[s["date"].isin(base)].groupby("time_of_day")["congestion_intensity"]
               .agg(baseline_mean="mean", baseline_std="std").reset_index(), on="time_of_day", how="inner")
    t = pd.to_datetime(prof["time_of_day"], format="%H:%M")
    prof["tod_min"] = t.dt.hour * 60 + t.dt.minute
    prof = prof.sort_values("tod_min").reset_index(drop=True)
    mean_pct = np.mean((prof["event"] - prof["baseline_mean"]) / (prof["baseline_mean"] + 1e-9)) * 100
    return prof, mean_pct


# =============================================================================
# Command line
# =============================================================================
def run_validation(benchmark_csv, slant_csv, yolo_csv, out_dir):
    counts, speeds = compare_methods(load_detections(benchmark_csv), load_detections(slant_csv), load_detections(yolo_csv))
    counts.to_csv(out_dir / "validation_counts_3min.csv", index=False)
    speeds.to_csv(out_dir / "validation_speeds_3min.csv", index=False)
    metrics = pd.DataFrame([count_metrics(counts, "slope"), count_metrics(counts, "yolo")])
    metrics.to_csv(out_dir / "validation_count_metrics.csv", index=False)
    print(metrics.round(3).to_string(index=False))
    print("Speed agreement (slant vs YOLO):", {k: round(v, 3) for k, v in speed_correlation(speeds).items()})


def run_events(out_dir, yolo_csvs):
    ts15 = network_timeseries(load_detections(*yolo_csvs), "15min")
    ts15.to_csv(out_dir / "network_timeseries_15min.csv", index=False)
    for name, ev, base in [("gameday", [config.GAMEDAY], config.GAMEDAY_BASELINE),
                           ("thanksgiving", config.THANKSGIVING, config.THANKSGIVING_BASELINE)]:
        prof, pct = event_diurnal_profile(ts15, ev, base)
        prof.to_csv(out_dir / f"network_profile_{name}.csv", index=False)
        print(f"{name}: 24-h mean change in congestion intensity {pct:+.1f}%")

        seg_df = load_segment_tables(ev + base)
        seg_change, _ = segment_event_change(seg_df, ev, base)
        seg_change.to_csv(out_dir / f"segment_change_{name}.csv", index=False)
        try:
            from geometry import export_segments
            export_segments(seg_change, ["pct_change_mean"], out_dir / f"segment_change_{name}.geojson")
        except Exception as e:  # geometry is optional
            print(f"GeoJSON skipped ({e})")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    v = sub.add_parser("validate", help="compare benchmark / slant stacking / YOLO on one labelled day")
    v.add_argument("--benchmark", required=True)
    v.add_argument("--slant", required=True)
    v.add_argument("--yolo", required=True)
    a30 = sub.add_parser("agreement", help="30-min agreement, slant stacking vs YOLO")
    a30.add_argument("--slant", required=True)
    a30.add_argument("--yolo", required=True)
    sub.add_parser("aggregate", help="segment tables for every day in RESULTS_DIR")
    e = sub.add_parser("events", help="network and segment event analysis")
    e.add_argument("--yolo", nargs="+", required=True, help="monthly YOLO speeds.csv files (Oct, Nov)")
    a = p.parse_args()

    out = config.ensure_dir(config.OUTPUT_DIR)
    if a.mode == "validate":
        run_validation(a.benchmark, a.slant, a.yolo, out)
    elif a.mode == "agreement":
        table, fits = agreement_30min(load_detections(a.slant), load_detections(a.yolo))
        table.to_csv(out / "agreement_30min.csv", index=False)
        fits.to_csv(out / "agreement_30min_fits.csv", index=False)
        print(fits.round(3).to_string(index=False))
    elif a.mode == "aggregate":
        aggregate_all_days()
    else:
        run_events(out, a.yolo)
