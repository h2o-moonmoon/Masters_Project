# pose_extract.py
import os, json, numpy as np, cv2
from ultralytics import YOLO  # pip install ultralytics
from pathlib import Path

VIDEO_DIR = Path("data/videos")
OUT_DIR = Path("data/poses"); OUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO("yolov8n-pose.pt")  # light; swap to s/m as needed

def extract_kpts(video_path, out_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kpts, confs = [], []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = model.predict(frame, verbose=False)[0]
        if len(res.keypoints) > 0:
            # take highest-conf person (assume single subject)
            idx = int(np.argmax(res.boxes.conf.cpu().numpy())) if len(res.boxes) else 0
            kp = res.keypoints.xy[idx].cpu().numpy()      # [J,2]
            kc = res.keypoints.conf[idx].cpu().numpy()    # [J]
        else:
            kp = np.full((17,2), np.nan); kc = np.zeros(17)
        kpts.append(kp); confs.append(kc); i += 1
    cap.release()
    np.savez_compressed(out_path, kpts=np.array(kpts), confs=np.array(confs), fps=fps, n_frames=frames)

if __name__ == "__main__":
    for mp4 in VIDEO_DIR.rglob("*.mp4"):
        out = OUT_DIR / (mp4.stem + ".npz")
        if not out.exists():
            extract_kpts(mp4, out)
