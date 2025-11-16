"""
Dataset prep CLI:
- (Optional) Pose extraction with YOLOv8-Pose (cached as .npz per video)
- Per-video rep segmentation (3 reps) using a robust motion signal from pose
- LOSO split generation (subject-level) + val split grouped by base event (multi-view safe)

Inputs:
  metadata.csv with columns: file, exercise, form, subtype, subject, camera
Video root:
  data/videos/
Pose cache:
  data/poses/

Outputs:
  data/rep_segments.json
  splits.json
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Optional: only import heavy deps when used
try:
    import cv2  # for reading video if needed
except Exception:
    cv2 = None

# pose libs are imported lazily inside functions
from scipy.signal import savgol_filter, find_peaks
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# ---------------------------
# Utility & validation
# ---------------------------

REQUIRED_COLS = ["file", "exercise", "form", "subtype", "subject", "camera"]

CAMERA_PATTERNS = [
    r"[_\-]?cam(?P<cam>\d+)",   # _cam3 / -cam2 / cam5
    r"[_\-]?c(?P<cam>\d+)",     # _c3 / -c2 / c5
    r"[_\-]?view(?P<cam>\d+)",  # _view1 etc.
]

def assert_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {missing}. "
                         f"Expected: {REQUIRED_COLS}")

def safe_stem(pathlike: str) -> str:
    return Path(pathlike).stem

def strip_camera_from_stem(stem: str, cam_value: str) -> str:
    """
    Build a base event id by removing the camera token from the filename stem.
    We try pattern-based removal first; if not present, try removing the literal cam value.
    """
    s = stem
    # 1) remove pattern tokens like _cam3/_c3/-cam3
    for pat in CAMERA_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)

    # 2) if a numeric camera value is provided, try removing trailing/isolated forms
    cam_clean = str(cam_value).strip()
    if cam_clean:
        # remove '_<cam>' or '-<cam>' or end-with-<cam>
        s = re.sub(rf"([_\-]?){re.escape(cam_clean)}($|[_\-])", r"\1", s, flags=re.IGNORECASE)

    # normalize multiple separators left behind
    s = re.sub(r"[_\-]{2,}", "_", s)
    return s.strip("_-")

def load_or_none(npz_path: Path):
    try:
        return np.load(npz_path)
    except Exception:
        return None


# ---------------------------
# Pose extraction (optional)
# ---------------------------

def extract_pose_for_video(video_path: Path, out_path: Path, model_name: str = "yolov8n-pose.pt"):
    """
    Extract 2D keypoints with Ultralytics YOLOv8-Pose and cache as .npz
    Arrays saved:
      - kpts: [T, J, 2]
      - confs: [T, J]
      - fps: float
      - n_frames: int
    """
    from ultralytics import YOLO

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)

    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    kpts_list, confs_list = [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = model.predict(frame, verbose=False)[0]

        if getattr(res, "keypoints", None) is not None and len(res.keypoints) > 0:
            # choose highest conf person (single subject assumption)
            if len(res.boxes) > 0:
                bconfs = res.boxes.conf.cpu().numpy()
                person_idx = int(np.argmax(bconfs))
            else:
                person_idx = 0
            kp = res.keypoints.xy[person_idx].cpu().numpy()       # [J, 2]
            kc = res.keypoints.conf[person_idx].cpu().numpy()     # [J]
        else:
            # if no detection, fill with NaNs and zeros (we'll smooth later)
            kp = np.full((17, 2), np.nan, dtype=np.float32)
            kc = np.zeros(17, dtype=np.float32)

        kpts_list.append(kp)
        confs_list.append(kc)
        idx += 1

    cap.release()

    kpts = np.array(kpts_list, dtype=np.float32)  # [T, J, 2]
    confs = np.array(confs_list, dtype=np.float32)  # [T, J]

    np.savez_compressed(out_path, kpts=kpts, confs=confs, fps=float(fps), n_frames=int(total))


# ---------------------------
# Rep segmentation from pose
# ---------------------------

# COCO-ish indices (adjust if your pose model uses different indices):
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12
PELVIS = 11  # use mid-hip fallback when pelvis is missing

def normalize_pose(kpts: np.ndarray, confs: np.ndarray) -> np.ndarray:
    """ Center at pelvis/mid-hip and scale by avg(shoulder_dist, hip_dist). """
    T, J, _ = kpts.shape
    # center at pelvis or mid-hip
    center = kpts[:, PELVIS, :]
    if np.isnan(center).any():
        center = np.nanmean(kpts[:, [L_HIP, R_HIP], :], axis=1)
    kp = kpts - center[:, None, :]

    sh = np.linalg.norm(kpts[:, L_SHO, :] - kpts[:, R_SHO, :], axis=1)
    hip = np.linalg.norm(kpts[:, L_HIP, :] - kpts[:, R_HIP, :], axis=1)
    scale = np.nanmedian((sh + hip) / 2.0)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return kp / scale

def motion_signal(kp: np.ndarray, confs: np.ndarray) -> np.ndarray:
    """ Confidence-weighted joint velocity magnitude, smoothed. """
    # mask non-finite joint coords
    finite = np.isfinite(kp).all(axis=-1) & np.isfinite(confs)
    kp = np.where(finite[:, :, None], kp, np.nan)

    # velocities
    vel = np.nan_to_num(np.diff(kp, axis=0), nan=0.0)  # [T-1, J, 2]
    vmag = np.linalg.norm(vel, axis=-1)                # [T-1, J]

    # confidence weights (clip to [0,1])
    cw = np.clip(confs[:-1], 0.0, 1.0)
    sig = (vmag * cw).sum(axis=1) / (cw.sum(axis=1) + 1e-6)  # [T-1]

    # smooth with Savitzky-Golay
    if len(sig) >= 11:
        win = min(31, (len(sig)//5)*2 + 1)
        win = max(win, 11)  # at least 11 and odd
        if win % 2 == 0: win += 1
        sig = savgol_filter(sig, win, polyorder=2, mode="interp")
    return sig

def find_three_reps(sig: np.ndarray, fps: float) -> List[Tuple[int, int]]:
    """ Return three [start,end] frame indices (based on signal peaks). """
    if len(sig) < 9:
        T = len(sig) + 1  # because sig is T-1 long
        return [(0, T//3), (T//3, 2*T//3), (2*T//3, T-1)]

    # peak detection
    min_dist = int(max(1, 0.6 * fps))  # min distance between peaks
    prominence = np.percentile(sig, 60) if len(sig) > 0 else 0.0
    peaks, props = find_peaks(sig, distance=min_dist, prominence=prominence)

    # relax if too few peaks
    if len(peaks) < 3:
        peaks, props = find_peaks(sig, distance=int(max(1, 0.4 * fps)))

    Tm1 = len(sig)
    if len(peaks) >= 3:
        # top-3 by prominence
        prom = props.get("prominences", np.ones_like(peaks, dtype=float))
        top3 = np.argsort(prom)[::-1][:3]
        peaks = np.sort(peaks[top3])

        bounds = [0]
        for i in range(len(peaks) - 1):
            bounds.append(int((peaks[i] + peaks[i + 1]) // 2))
        bounds.append(Tm1 - 1)
        segs = []
        for i in range(3):
            s, e = bounds[i], bounds[i + 1]
            # small padding
            s = max(0, s - 5)
            e = min(Tm1 - 1, e + 5)
            segs.append((s, e))
        return segs
    else:
        # equal thirds fallback
        T = Tm1 + 1
        return [(0, T//3), (T//3, 2*T//3), (2*T//3, T-1)]

def segment_one_npz(npz_path: Path) -> List[Tuple[int, int]]:
    data = np.load(npz_path)
    kpts, confs = data["kpts"], data["confs"]
    fps = float(data["fps"])
    if kpts.shape[0] < 10:
        T = kpts.shape[0]
        return [(0, T//3), (T//3, 2*T//3), (2*T//3, T-1)]
    kp_n = normalize_pose(kpts, confs)
    sig = motion_signal(kp_n, confs)
    segs = find_three_reps(sig, fps)
    # convert to original frame indices (sig is T-1 long)
    T = kpts.shape[0]
    segs = [(max(0, s), min(T - 1, e)) for s, e in segs]
    return segs


# ---------------------------
# Splits (LOSO + val by event)
# ---------------------------

def build_event_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create event_id (same base capture across cameras).
    We remove camera token from the filename stem using the known camera column.
    """
    stems = df["file"].apply(lambda x: safe_stem(x))
    event_ids = []
    for stem, cam in zip(stems, df["camera"].astype(str).tolist()):
        event_ids.append(strip_camera_from_stem(stem, cam))
    df = df.copy()
    df["stem"] = stems
    df["event_id"] = event_ids
    return df

def make_splits(df: pd.DataFrame, n_val_splits: int = 5) -> Dict[str, Any]:
    """
    LOSO by 'subject'. Validation split is a group split by 'event_id'
    within the training subjects to avoid multi-view leakage.
    """
    subjects = sorted(df["subject"].unique().tolist())
    if len(subjects) < 2:
        raise ValueError("Need at least 2 unique subjects for LOSO.")

    folds = []
    for test_sid in subjects:
        test_mask = (df["subject"] == test_sid)
        test_df = df[test_mask]
        trainval_df = df[~test_mask]

        n_groups = trainval_df["event_id"].nunique()
        k = min(max(3, n_val_splits), n_groups)  # keep sensible k

        # GroupKFold on event_id
        gkf = GroupKFold(n_splits=k)
        # Use the first split deterministically
        tr_idx, val_idx = next(gkf.split(trainval_df, groups=trainval_df["event_id"]))

        tr_files = trainval_df.iloc[tr_idx]["file"].tolist()
        val_files = trainval_df.iloc[val_idx]["file"].tolist()
        test_files = test_df["file"].tolist()

        folds.append({
            "test_subject": str(test_sid),
            "train_files": tr_files,
            "val_files": val_files,
            "test_files": test_files
        })

    return {"folds": folds}


# ---------------------------
# Orchestration
# ---------------------------

def run_cli(
    metadata_csv: Path,
    video_root: Path,
    pose_root: Path,
    segments_out: Path,
    splits_out: Path,
    model_name: str,
    do_pose: bool,
    n_val_splits: int,
    overwrite_segments: bool,
):
    # Load metadata
    df = pd.read_csv(metadata_csv)
    assert_columns(df)
    df = build_event_ids(df)

    # Pose extraction (optional or on-demand)
    if do_pose:
        try:
            from ultralytics import YOLO  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "Ultralytics is required for --extract-pose. "
                "Install with: pip install ultralytics"
            ) from e

        print("[Pose] Extracting pose keypoints (cached)...")
        for _, row in df.iterrows():
            vid_path = video_root / row["file"]
            npz_path = pose_root / (safe_stem(row["file"]) + ".npz")
            if not npz_path.exists():
                extract_pose_for_video(vid_path, npz_path, model_name=model_name)

    # Rep segmentation
    if segments_out.exists() and not overwrite_segments:
        print(f"[Segmentation] {segments_out} exists; skipping (use --overwrite-segments to regenerate).")
        with open(segments_out, "r") as f:
            segments = json.load(f)
    else:
        print("[Segmentation] Computing 3-rep segments...")
        segments = {}
        missing_pose = 0
        for _, row in df.iterrows():
            stem = safe_stem(row["file"])
            npz_path = pose_root / f"{stem}.npz"
            if not npz_path.exists():
                if do_pose:
                    # should've been created; but if not, try now
                    vid_path = video_root / row["file"]
                    extract_pose_for_video(vid_path, npz_path, model_name=model_name)
                else:
                    missing_pose += 1
                    continue
            try:
                segs = segment_one_npz(npz_path)
                segments[stem] = {"segments": segs}
            except Exception as e:
                print(f"[WARN] Segmentation failed for {stem}: {e}")
        if missing_pose and not do_pose:
            print(f"[WARN] {missing_pose} videos missing pose cache. "
                  f"Run with --extract-pose to generate.")
        segments_out.parent.mkdir(parents=True, exist_ok=True)
        with open(segments_out, "w") as f:
            json.dump(segments, f, indent=2)
        print(f"[Segmentation] Wrote {segments_out} ({len(segments)} entries)")

    # Splits (LOSO + val by event)
    print("[Splits] Building LOSO + grouped val splits...")
    splits = make_splits(df, n_val_splits=n_val_splits)
    with open(splits_out, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[Splits] Wrote {splits_out} ({len(splits['folds'])} folds)")

    # Quick diagnostics
    # Check for multi-view integrity: each event_id should map to multiple cameras ideally
    event_sizes = df.groupby("event_id")["camera"].nunique()
    n_multi = int((event_sizes >= 2).sum())
    n_total = event_sizes.shape[0]
    print(f"[Info] Event groups with ≥2 cameras: {n_multi}/{n_total}")

    # Check that no event_id leaks across train/test per fold
    leaks = 0
    for fi, fold in enumerate(splits["folds"], 1):
        tr_ev = set(df[df["file"].isin(fold["train_files"])]["event_id"])
        te_ev = set(df[df["file"].isin(fold["test_files"])]["event_id"])
        inter = tr_ev & te_ev
        if inter:
            leaks += 1
            print(f"[WARN] Fold {fi} has event leakage across train/test for events: {sorted(list(inter))[:5]} ...")
    if leaks == 0:
        print("[Check] No train/test event leakage detected across folds.")

def main():
    p = argparse.ArgumentParser(description="Prepare pose, rep segments, and LOSO splits.")
    p.add_argument("--metadata", type=Path, default=Path("metadata.csv"), help="Path to metadata CSV.")
    p.add_argument("--video-dir", type=Path, default=Path("data/videos"), help="Root directory containing videos.")
    p.add_argument("--pose-dir", type=Path, default=Path("data/poses"), help="Directory to cache pose npz files.")
    p.add_argument("--segments-out", type=Path, default=Path("data/rep_segments.json"), help="Output JSON for rep segments.")
    p.add_argument("--splits-out", type=Path, default=Path("splits.json"), help="Output JSON for LOSO splits.")
    p.add_argument("--extract-pose", action="store_true", help="Extract pose keypoints for all videos before segmentation.")
    p.add_argument("--pose-model", type=str, default="yolov8n-pose.pt", help="Ultralytics pose model (e.g., yolov8n-pose.pt).")
    p.add_argument("--val-splits", type=int, default=5, help="Number of GroupKFold splits for validation within train subjects.")
    p.add_argument("--overwrite-segments", action="store_true", help="Regenerate segments even if JSON exists.")
    args = p.parse_args()

    run_cli(
        metadata_csv=args.metadata,
        video_root=args.video_dir,
        pose_root=args.pose_dir,
        segments_out=args.segments_out,
        splits_out=args.splits_out,
        model_name=args.pose_model,
        do_pose=args.extract_pose,
        n_val_splits=args.val_splits,
        overwrite_segments=args.overwrite_segments,
    )

if __name__ == "__main__":
    main()
