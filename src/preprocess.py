# src/preprocess.py
import os
import math
import argparse
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

# --------- Defaults ------------------------
DEFAULT_IN_FPS = 30        # Target FPS for output videos
DEFAULT_SHORT_SIDE = 480   # Resize so the shorter side == this (keeps aspect)
DEFAULT_CODEC = "mp4v"     # Use mp4v for wide compatibility
# -------------------------------------------

def make_even(x: int) -> int:
    """Make dimension even (some codecs need even width/height)."""
    return x if x % 2 == 0 else x - 1

def probe_size_fps(cap):
    """Get width, height, fps from an opened cv2.VideoCapture."""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    # Some files report 0 fps; fallback
    if fps <= 1e-3:
        fps = DEFAULT_IN_FPS
    return w, h, fps

def standardize_video(in_path: Path, out_path: Path, out_fps=DEFAULT_IN_FPS, short_side=DEFAULT_SHORT_SIDE, codec=DEFAULT_CODEC) -> bool:
    """Read a video, resize to keep aspect with given short_side, force even dims, set fps."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {in_path}")
        return False

    w, h, _ = probe_size_fps(cap)
    if w == 0 or h == 0:
        print(f"[WARN] Invalid size for: {in_path}")
        cap.release()
        return False

    scale = short_side / min(w, h)
    new_w = make_even(int(round(w * scale)))
    new_h = make_even(int(round(h * scale)))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (new_w, new_h))
    if not writer.isOpened():
        print(f"[WARN] Cannot open writer for: {out_path}")
        cap.release()
        return False

    # Frame timing: read all, write at fixed out_fps
    # (We just re-encode; if you want exact frame dropping/duplication logic, add it later.)
    frames_written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        writer.write(resized)
        frames_written += 1

    cap.release()
    writer.release()
    if frames_written == 0:
        print(f"[WARN] No frames written for: {in_path}")
        return False
    return True

def derive_out_path(video_path: str, standardized_root: Path) -> Path:
    """
    Keep the same relative structure under outputs/standardized.
    Example:
        data/raw_videos/squat/squat_001_hi.mp4
      -> outputs/standardized/squat/squat_001_hi.mp4
    """
    p = Path(video_path)
    # Try to drop the leading 'data/raw_videos' if present
    try:
        idx = p.parts.index("raw_videos")
        rel = Path(*p.parts[idx+1:])  # e.g., squat/squat_001_hi.mp4
    except ValueError:
        # If no 'raw_videos' in path, just use the filename and parent name if present
        rel = p.name if p.parent == p.anchor else Path(p.name)
    out_path = standardized_root / rel
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Standardize videos (FPS + resize short side) and write updated CSV.")
    parser.add_argument("--labels", type=str, default="data/annotations/labels.csv",
                        help="Path to labels CSV with a 'video_path' column.")
    parser.add_argument("--out-root", type=str, default="outputs/standardized",
                        help="Root folder to place standardized videos under.")
    parser.add_argument("--out-csv", type=str, default="data/annotations/labels_standardized.csv",
                        help="Path to write updated CSV with standardized paths.")
    parser.add_argument("--inplace", action="store_true",
                        help="Replace 'video_path' with standardized paths instead of adding a new column.")
    parser.add_argument("--fps", type=float, default=DEFAULT_IN_FPS,
                        help="Target FPS for output videos.")
    parser.add_argument("--short-side", type=int, default=DEFAULT_SHORT_SIDE,
                        help="Shorter side will be resized to this value, keeping aspect ratio.")
    parser.add_argument("--codec", type=str, default=DEFAULT_CODEC,
                        help="FourCC codec, e.g., mp4v, avc1, H264 (depends on build).")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    out_root = Path(args.out_root)
    out_csv = Path(args.out_csv)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")

    df = pd.read_csv(labels_path, sep='\t')
    df.columns = df.columns.str.strip().str.lower().str.replace('\ufeff', '')
    print(df.columns.tolist())

    if "video_path" not in df.columns:
        raise ValueError("labels.csv must contain a 'video_path' column.")

    standardized_paths = []
    successes = 0
    failures = 0

    print(f"[INFO] Standardizing {len(df)} videos → {out_root}")
    for _, row in tqdm(df.iterrows(), total=len(df), ncols=80):
        vpath = Path(str(row["video_path"]))
        if not vpath.exists():
            # Try relative to project root
            alt = Path(".") / vpath
            if alt.exists():
                vpath = alt
            else:
                print(f"[WARN] Missing video: {vpath}")
                standardized_paths.append("")
                failures += 1
                continue

        out_path = derive_out_path(str(vpath), out_root)
        ok = standardize_video(vpath, out_path, out_fps=args.fps, short_side=args.short_side, codec=args.codec)
        if ok:
            standardized_paths.append(str(out_path.as_posix()))
            successes += 1
        else:
            standardized_paths.append("")
            failures += 1

    if args.inplace:
        df["video_path"] = standardized_paths
    else:
        df["standardized_video_path"] = standardized_paths

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("\n[DONE]")
    print(f"  Successful: {successes}")
    print(f"  Failed:     {failures}")
    print(f"  Wrote CSV:  {out_csv}")

if __name__ == "__main__":
    main()
