# src/extract_keypoints.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import torch

# Save (T, 17, 3): x, y, conf per keypoint (conf is safe even if model doesn’t provide kp conf)
DEFAULT_MODEL = "yolov8n-pose.pt"
CONF_THRESH = 0.25

def open_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        # YOLO expects RGB
        yield frame_bgr[:, :, ::-1]
    cap.release()

def select_main_person(result):
    """
    Given a single Ultralytics Result, pick the person with the highest box confidence.
    Returns (xy: (17,2) np.ndarray, conf_kp: (17,) np.ndarray) OR (None, None) if nothing found.
    """
    # No detections at all
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None, None

    # Boxes confidences (shape: N,)
    boxes_conf = result.boxes.conf
    if boxes_conf is None or len(boxes_conf) == 0:
        return None, None

    # Keypoints xy: (N, 17, 2)
    kps_xy = result.keypoints.xy
    if kps_xy is None or len(kps_xy) == 0:
        return None, None

    # Pick index of the most confident person
    idx = int(torch.argmax(boxes_conf).item())

    # Extract keypoints xy
    xy = kps_xy[idx]  # tensor (17,2)
    xy = xy.detach().cpu().numpy().astype(np.float32)

    # Try to get per-kp conf; some versions/models return None
    kp_conf = result.keypoints.conf
    if kp_conf is not None:
        conf_vec = kp_conf[idx].detach().cpu().numpy().astype(np.float32)  # (17,)
    else:
        # Fallback: use the person (box) confidence for all keypoints
        person_conf = float(boxes_conf[idx].detach().cpu().item())
        conf_vec = np.full((xy.shape[0],), person_conf, dtype=np.float32)

    return xy, conf_vec

def extract_sequence(model, video_path, conf=CONF_THRESH, max_frames=None):
    """
    Returns np.ndarray of shape (T, 17, 3) with (x,y,conf).
    If no person is found in a frame, inserts zeros (17,3).
    """
    seq = []
    frame_iter = open_video_frames(video_path)

    for i, img_rgb in enumerate(frame_iter):
        if max_frames is not None and i >= max_frames:
            break
        # Single-frame predict (stream=True avoids big memory spike)
        results = model.predict(img_rgb, conf=conf, verbose=False)
        if not results:
            seq.append(np.zeros((17, 3), dtype=np.float32))
            continue

        res = results[0]
        xy, conf_vec = select_main_person(res)
        if xy is None:
            seq.append(np.zeros((17, 3), dtype=np.float32))
        else:
            seq.append(np.hstack([xy, conf_vec[:, None]]))  # (17,3)

    if len(seq) == 0:
        return None
    return np.stack(seq, axis=0).astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Extract YOLOv8-pose keypoints to .npy sequences.")
    parser.add_argument("--labels", type=str, default="data/annotations/labels.csv",
                        help="CSV with at least a 'video_path' column.")
    parser.add_argument("--outdir", type=str, default="outputs/keypoints",
                        help="Directory to write .npy sequences.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Ultralytics pose model (e.g., yolov8n-pose.pt, yolov8s-pose.pt).")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cpu' or 'cuda'. By default Ultralytics will choose.")
    parser.add_argument("--conf", type=float, default=CONF_THRESH,
                        help="Detection confidence threshold.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional cap on frames per video for quick tests.")
    parser.add_argument("--path-col", type=str, default="video_path",
                        help="Column name in CSV that contains the video path.")
    args = parser.parse_args()

    df = pd.read_csv(args.labels, sep='\t')
    if args.path_col not in df.columns:
        raise ValueError(f"CSV must contain a '{args.path_col}' column.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load YOLO model
    model = YOLO(args.model)
    if args.device:
        model.to(args.device)

    for _, row in tqdm(df.iterrows(), total=len(df), ncols=80, desc="Extracting"):
        vpath = Path(str(row[args.path_col]))
        if not vpath.exists():
            print(f"[WARN] Missing video: {vpath}")
            continue

        seq = extract_sequence(model, vpath, conf=args.conf, max_frames=args.max_frames)
        if seq is None:
            print(f"[WARN] No frames read from: {vpath}")
            continue

        # Save under stem name
        out_path = outdir / (vpath.stem + ".npy")
        np.save(out_path, seq)

if __name__ == "__main__":
    main()
