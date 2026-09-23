"""
Deep-learning vehicle-trajectory detection on DAS spatiotemporal images.

    split_dataset()          split labelled real images into train / val / test (YOLO layout)
    train_model()            train one detector (YOLOv8 / YOLO11 / RT-DETR) with Ultralytics
    train_three_stage()      Stage 1 synthetic -> Stage 2 synthetic + real -> Stage 3 real fine-tune
    training_summary()       final loss / mAP / precision / recall of every run (from results.csv)
    evaluate_holdout()       evaluate every trained run on the independent hold-out set

Run ``python detection.py split|train|summary|holdout``.
"""
import argparse
import glob
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import config

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CLASS_NAMES = {0: "vehicle_trace"}

# Per-architecture hyper-parameters used in the paper
MODELS = {
    "yolov8s": dict(weights="yolov8s.pt", batch=8, lr0=0.003, close_mosaic=20, prefix="y8"),
    "yolo11s": dict(weights="yolo11s.pt", batch=8, lr0=0.003, close_mosaic=20, prefix="y11"),
    "rtdetr": dict(weights="rtdetr-l.pt", batch=4, lr0=0.0003, close_mosaic=0, prefix="rtd"),
}
EPOCHS = 20
PATIENCE = 10
WORKERS = 4
STAGES = [("stage1_synth", "SYNTH_YAML"), ("stage2_mix", "MIX_YAML"), ("stage3_real_ft", "REAL_YAML")]


# =============================================================================
# Dataset
# =============================================================================
def _is_negative(label_file):
    """Empty (or whitespace-only) label file => image without vehicles."""
    with open(label_file, encoding="utf-8") as f:
        return not any(line.strip() for line in f)


def _place(images, split, label_dir, out_root, copy=True):
    for img in images:
        lbl = Path(label_dir) / (Path(img).stem + ".txt")
        for src, dst in [(img, Path(out_root) / "images" / split / Path(img).name),
                         (lbl, Path(out_root) / "labels" / split / lbl.name)]:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            if not Path(src).exists():          # missing label -> empty txt
                dst.touch()
            elif copy:
                shutil.copy2(src, dst)
            else:
                os.link(src, dst)               # hard link (same drive only)


def write_yaml(path, root, train="images/train", val="images/val", test="images/test"):
    names = "\n".join(f"  {k}: {v}" for k, v in CLASS_NAMES.items())
    Path(path).write_text(f"path: {root}\ntrain: {train}\nval: {val}\ntest: {test}\n\nnames:\n{names}\n",
                          encoding="utf-8")
    return Path(path)


def split_dataset(img_dir=config.LABELED_IMG_DIR, lbl_dir=config.LABELED_LBL_DIR,
                  out_root=config.DATASET_SPLIT_DIR, pos_split=(0.6, 0.2, 0.2),
                  val_neg_frac=0.2, test_neg_frac=0.2, seed=42, copy=True):
    """
    Split positive images by ``pos_split``; add negatives (empty labels) to val and
    test at ``*_neg_frac`` of their positive count, the rest go to train.
    Writes ``dataset_real.yaml`` in ``out_root``.

    The revised manuscript uses 1,200 real images split 3:1 into train / val
    (``pos_split=(0.75, 0.25, 0.0)``) with a separate 500-image hold-out set.
    """
    random.seed(seed)
    imgs = sorted(p for ext in IMG_EXTS for p in glob.glob(os.path.join(img_dir, f"*{ext}")))
    if not imgs:
        raise FileNotFoundError(f"No images in {img_dir}")

    pos, neg, missing = [], [], []
    for img in imgs:
        lbl = Path(lbl_dir) / (Path(img).stem + ".txt")
        (missing if not lbl.exists() else neg if _is_negative(lbl) else pos).append(img)
    print(f"positive={len(pos)} | negative={len(neg)} | missing label={len(missing)}")
    if missing:
        config.ensure_dir(out_root)
        (Path(out_root) / "missing_label_images.txt").write_text("\n".join(missing), encoding="utf-8")

    random.shuffle(pos)
    n_tr, n_va = int(len(pos) * pos_split[0]), int(len(pos) * pos_split[1])
    train_pos, val_pos, test_pos = pos[:n_tr], pos[n_tr:n_tr + n_va], pos[n_tr + n_va:]

    random.shuffle(neg)
    n_vn, n_tn = int(len(val_pos) * val_neg_frac), int(len(test_pos) * test_neg_frac)
    val_neg, test_neg, train_neg = neg[:n_vn], neg[n_vn:n_vn + n_tn], neg[n_vn + n_tn:]

    for split, items in [("train", train_pos + train_neg), ("val", val_pos + val_neg), ("test", test_pos + test_neg)]:
        _place(items, split, lbl_dir, out_root, copy=copy)
        print(f"{split}: {len(items)}")
    return write_yaml(Path(out_root) / "dataset_real.yaml", out_root)


# =============================================================================
# Training
# =============================================================================
def train_model(arch, data_yaml, run_name, weights=None, seed=50, epochs=EPOCHS, project=config.RUNS_DIR):
    """Train one detector; returns the path of ``best.pt``."""
    from ultralytics import YOLO

    cfg = MODELS[arch]
    weights = weights or str(config.PRETRAINED_DIR / cfg["weights"])
    config.ensure_dir(project)
    YOLO(weights).train(
        data=str(data_yaml), imgsz=config.IMG_SIZE, batch=cfg["batch"], epochs=epochs, lr0=cfg["lr0"],
        seed=seed, device=config.DEVICE, workers=WORKERS, patience=PATIENCE,
        close_mosaic=cfg["close_mosaic"], project=str(project), name=run_name, cache=False,
    )
    return Path(project) / run_name / "weights" / "best.pt"


def train_three_stage(arch, seed=50):
    """
    Hybrid synthetic-real training. Each stage starts from the best weights of
    the previous one (Stage 1 starts from the COCO-pretrained checkpoint).
    Run names follow ``<prefix>_<stage>``, e.g. ``y8_stage2_mix``.
    """
    weights = None
    for stage, yaml_attr in STAGES:
        weights = str(train_model(arch, getattr(config, yaml_attr), f"{MODELS[arch]['prefix']}_{stage}",
                                  weights=weights, seed=seed))
    return weights


def train_real_only(arch, seed=50):
    """Baseline without synthetic pre-training (run name ``<arch>_seed<seed>``)."""
    return train_model(arch, config.REAL_YAML, f"{arch}_seed{seed}", seed=seed)


# =============================================================================
# Evaluation
# =============================================================================
def _first_col(df, keys):
    return next((k for k in keys if k in df.columns), None)


def training_summary(runs_dir=config.RUNS_DIR):
    """Final-epoch training loss and validation metrics of every run in ``runs_dir``."""
    rows = []
    for csv in sorted(Path(runs_dir).glob("*/results.csv")):
        df = pd.read_csv(csv)
        df.columns = df.columns.str.strip()
        loss = _first_col(df, ["train/box_loss", "train/cls_loss", "train/dfl_loss", "train/giou_loss",
                               "train/l1_loss", "train/loss"]) \
            or next((c for c in df.columns if c.startswith("train/") and "loss" in c.lower()), None)
        last = df.iloc[-1]
        rows.append({
            "run": csv.parent.name,
            "epochs": len(df),
            "loss_key": loss,
            "final_loss": last[loss] if loss else np.nan,
            "mAP50-95": last.get(_first_col(df, ["metrics/mAP50-95(B)", "metrics/mAP50-95"]), np.nan),
            "precision": last.get(_first_col(df, ["metrics/precision(B)", "metrics/precision"]), np.nan),
            "recall": last.get(_first_col(df, ["metrics/recall(B)", "metrics/recall"]), np.nan),
        })
    return pd.DataFrame(rows)


def find_best_weights(runs_dir=config.RUNS_DIR):
    return {p.parent.parent.name: str(p) for p in sorted(Path(runs_dir).glob("*/weights/best.pt"),
                                                         key=lambda p: p.parent.parent.name.lower())}


def evaluate_holdout(holdout_dir=config.HOLDOUT_DIR, runs_dir=config.RUNS_DIR, out_dir=None,
                     conf=0.001, iou=0.7):
    """Validate every ``best.pt`` on the hold-out set; returns P, R, mAP50, mAP50-95 per run."""
    from ultralytics import YOLO

    out_dir = config.ensure_dir(out_dir or Path(runs_dir) / "holdout_eval")
    data = write_yaml(out_dir / "holdout_test.yaml", holdout_dir, train="images", val="images", test="images")
    rows = []
    for tag, w in find_best_weights(runs_dir).items():
        print(f"=== Evaluating {tag} ===")
        r = YOLO(w).val(data=str(data), imgsz=config.IMG_SIZE, conf=conf, iou=iou, device=config.DEVICE,
                        split="test", plots=True, project=str(out_dir), name=tag)
        rd = getattr(r, "results_dict", {}) or {}
        box = getattr(r, "box", None)
        rows.append({
            "tag": tag, "weight": w,
            "P": getattr(box, "mp", rd.get("metrics/precision(B)")),
            "R": getattr(box, "mr", rd.get("metrics/recall(B)")),
            "mAP50": getattr(box, "map50", rd.get("metrics/mAP50(B)")),
            "mAP50-95": getattr(box, "map", rd.get("metrics/mAP50-95(B)")),
        })
    df = pd.DataFrame(rows).sort_values(["mAP50-95", "mAP50"], ascending=False, na_position="last")
    df.to_csv(out_dir / "holdout_metrics_summary.csv", index=False)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["split", "train", "summary", "holdout"])
    p.add_argument("--arch", nargs="+", default=list(MODELS), choices=list(MODELS))
    a = p.parse_args()
    if a.step == "split":
        split_dataset()
    elif a.step == "train":
        for arch in a.arch:
            train_three_stage(arch)
            train_real_only(arch)
    elif a.step == "summary":
        out = config.ensure_dir(config.OUTPUT_DIR) / "training_summary.csv"
        training_summary().to_csv(out, index=False)
        print(f"Saved {out}")
    else:
        print(evaluate_holdout())
