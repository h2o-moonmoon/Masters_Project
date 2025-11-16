# src/datasets/core.py
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import random
import math

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tvT
import torchvision.transforms.functional as tvF

try:
    import decord
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False

# ---------------------------------------------------------
# Resolve paths relative to project root (parent of src/)
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # project/
DATA_DIR = ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
POSES_DIR = DATA_DIR / "poses"
META_CSV = DATA_DIR / "metadata.csv"
SEG_JSON = DATA_DIR / "rep_segments.json"
SPLITS_JSON = ROOT / "splits.json"

# ---------------------------------------------------------
# Label encoders (exercise, form, subtype)
# ---------------------------------------------------------
def build_label_maps(df: pd.DataFrame) -> Dict[str, Dict[Any, int]]:
    """
    Creates stable id maps:
      exercise2id: categorical (5 classes)
      form2id: {'correct','incorrect'} -> {0,1}
      subtype2id: integers as in your CSV (0,1,2) mapped to themselves
    """
    exercises = sorted(df["exercise"].unique().tolist())
    exercise2id = {ex: i for i, ex in enumerate(exercises)}

    # form is typically 'correct'/'incorrect'
    forms = sorted(df["form"].unique().tolist())
    form2id = {f: i for i, f in enumerate(forms)}  # ensures stable

    # subtype already numeric per your schema (0,1,2)
    subtypes = sorted(df["subtype"].unique().tolist())
    subtype2id = {int(s): int(s) for s in subtypes}

    return dict(exercise2id=exercise2id, form2id=form2id, subtype2id=subtype2id)


# ---------------------------------------------------------
# Frame sampling helpers
# ---------------------------------------------------------
def uniform_indices(num_frames: int, req: int) -> np.ndarray:
    """
    Evenly sample 'req' indices from [0, num_frames-1].
    """
    if req <= 1 or num_frames <= 1:
        return np.array([min(0, num_frames - 1)], dtype=np.int32)
    idx = np.linspace(0, num_frames - 1, num=req, dtype=np.int32)
    return idx

def clip_indices(start: int, end: int, req: int) -> np.ndarray:
    """
    Uniformly sample 'req' indices from [start, end] inclusive.
    """
    total = max(1, end - start + 1)
    rel = np.linspace(0, total - 1, num=req, dtype=np.int32)
    return start + rel


# ---------------------------------------------------------
# Video reading (OpenCV)
# ---------------------------------------------------------
def read_video_frames(path: Path, frame_indices: np.ndarray, backend: str = "auto") -> List[np.ndarray]:
    """
    Returns list of BGR frames.
    backend: "auto" | "decord" | "opencv"
    """
    if backend == "decord" or (backend == "auto" and _HAS_DECORD):
        vr = decord.VideoReader(str(path))
        frames = [vr[int(i)].asnumpy() for i in frame_indices]
        # Decord returns RGB; convert to BGR for consistency with rest of code
        frames = [f[:, :, ::-1].copy() for f in frames]
        return frames

    # fallback: OpenCV
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames = []
    last_pos = -1
    for fi in frame_indices:
        if fi != last_pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            frames.append(frame)
        last_pos = fi
    cap.release()
    return frames



# ---------------------------------------------------------
# Simple video transforms (augment/no-augment)
# ---------------------------------------------------------
class VideoTransform:
    """
    Apply torchvision-like transforms to a list of frames.
    We’ll keep it simple & consistent across models.
    """
    def __init__(self, size: int = 224, augment: bool = False, allow_hflip: bool = False):
        self.size = size
        self.augment = augment
        self.allow_hflip = allow_hflip
        # base transforms
        self.resize = tvT.Resize((size, size))
        self.to_tensor = tvT.ToTensor()
        self.normalize = tvT.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        # augmentation ops
        # self.color = tvT.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        self.color = tvT.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
        self.blur = tvT.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))

    def __call__(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        # BGR -> RGB
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        # PIL expects HWC RGB uint8; torchvision ops can handle tensors too, but we’ll go via F
        tensor_list = []
        do_hflip = self.allow_hflip and self.augment and (random.random() < 0.5)
        for img in frames_rgb:
            pil = tvF.to_pil_image(img).convert("RGB")
            if self.augment:
                # brightness/contrast/saturation are safe everywhere
                try:
                    pil = tvT.functional.adjust_brightness(pil, 1.0 + random.uniform(-0.2, 0.2))
                    pil = tvT.functional.adjust_contrast(pil, 1.0 + random.uniform(-0.2, 0.2))
                    pil = tvT.functional.adjust_saturation(pil, 1.0 + random.uniform(-0.2, 0.2))
                    # Hue jitter (guarded): only apply small hue and catch platform bugs
                    hue_factor = random.uniform(-0.05, 0.05)
                    if abs(hue_factor) > 0:
                        try:
                            pil = tvT.functional.adjust_hue(pil, hue_factor)
                        except Exception:
                            # skip hue on platforms that error
                            pass
                except Exception:
                    # If any color op fails, skip augmentation for this frame
                    pass

                if random.random() < 0.2:
                    pil = self.blur(pil)
            if do_hflip:
                pil = tvF.hflip(pil)

            pil = self.resize(pil)
            t = self.to_tensor(pil)
            t = self.normalize(t)
            tensor_list.append(t)
        # stack to [T, C, H, W]
        return torch.stack(tensor_list, dim=0)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class VideoDataset(Dataset):
    """
    Core dataset supporting:
      - per-video vs per-rep
      - augmentation on/off
      - reading labels (exercise/form/subtype)
      - optional pose npz path (you can read it here if you want keypoints)
    """
    def __init__(
        self,
        split_files: List[str],
        metadata_csv: Path = META_CSV,
        videos_dir: Path = VIDEOS_DIR,
        segments_json: Path = SEG_JSON,
        mode: str = "video",                   # 'video' or 'rep'
        num_frames: int = 32,
        resize: int = 224,
        augment: bool = False,
        allow_hflip: bool = False,
        include_pose: bool = False,
    ):
        """
        split_files: list of file paths exactly as in metadata (relative to project root)
        """
        super().__init__()
        self.videos_dir = videos_dir
        self.mode = mode
        assert self.mode in ("video", "rep")
        self.num_frames = num_frames
        self.include_pose = include_pose
        self._lengths = {}

        # load metadata
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df["file"].isin(split_files)].copy()
        if len(self.df) == 0:
            raise ValueError("No rows matched split list in metadata.")
        # label maps
        self.label_maps = build_label_maps(self.df)

        # segments (optional for per-rep)
        self.segments = {}
        if segments_json.exists():
            with open(segments_json, "r") as f:
                self.segments = json.load(f)

        # build index
        self.samples = self._build_index()

        # transforms
        self.vT = VideoTransform(size=resize, augment=augment, allow_hflip=allow_hflip)

    def _build_index(self) -> List[Dict[str, Any]]:
        samples = []
        for _, row in self.df.iterrows():
            rel = row["file"]
            path = VIDEOS_DIR / rel
            stem = Path(rel).stem

            base = dict(
                path=path,
                stem=stem,
                exercise_id=self.label_maps["exercise2id"][row["exercise"]],
                form_id=self.label_maps["form2id"][row["form"]],
                subtype_id=int(row["subtype"]),
                subject=str(row.get("subject", "")),
                camera=str(row.get("camera", "")),
            )

            if self.mode == "video":
                # one sample per video
                samples.append({**base, "segment": None})
            else:
                # three samples per video using precomputed segments
                seg_info = self.segments.get(stem)
                if seg_info is None:
                    # fallback: equal thirds
                    # (we don’t know T here, so we’ll sample uniformly over full length at load time)
                    for k in range(3):
                        samples.append({**base, "segment": None, "rep_index": k})
                else:
                    segs = seg_info.get("segments", [])
                    for k, (s, e) in enumerate(segs):
                        samples.append({**base, "segment": (int(s), int(e)), "rep_index": k})
        return samples

    def __len__(self):
        return len(self.samples)

    def _pick_indices(self, total_frames: int, segment: Optional[Tuple[int, int]]) -> np.ndarray:
        if segment is None:
            # full video uniform
            return uniform_indices(total_frames, self.num_frames)
        else:
            s, e = segment
            s = max(0, min(s, total_frames - 1))
            e = max(0, min(e, total_frames - 1))
            if e < s: e = s
            if (e - s + 1) >= self.num_frames:
                return clip_indices(s, e, self.num_frames)
            else:
                # pad by repeating edges
                idx = clip_indices(s, e, e - s + 1)
                pad = np.pad(idx, (0, self.num_frames - len(idx)), mode="edge")
                return pad

    def _get_total_frames(self, path: Path) -> int:
        if path in self._lengths:
            return self._lengths[path]
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self._lengths[path] = total
        return total

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item = self.samples[i]
        path: Path = item["path"]
        stem = item["stem"]

        total = self._get_total_frames(path)
        indices = self._pick_indices(total, item.get("segment"))
        # frames = read_video_frames(path, indices)
        frames = read_video_frames(path, indices, backend="auto")

        # transforms -> tensor [T,C,H,W]
        video = self.vT(frames)  # float tensor normalized

        sample = {
            "video": video,                              # [T,C,H,W]
            "exercise_id": torch.tensor(item["exercise_id"], dtype=torch.long),
            "form_id": torch.tensor(item["form_id"], dtype=torch.long),
            "subtype_id": torch.tensor(item["subtype_id"], dtype=torch.long),
            "file_stem": stem,
            "subject": item["subject"],
            "camera": item["camera"],
            "indices": torch.from_numpy(indices.copy()).long(),
        }

        if self.include_pose:
            npz_path = POSES_DIR / f"{stem}.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                # You can return pose for future RNN models; here just returning shapes to avoid big memory by default
                sample["pose_meta"] = {
                    "fps": float(data["fps"]),
                    "n_frames": int(data["n_frames"]),
                }
            else:
                sample["pose_meta"] = None

        return sample


# ---------------------------------------------------------
# Split utilities
# ---------------------------------------------------------
def load_splits(splits_path: Path = SPLITS_JSON) -> Dict[str, List[str]]:
    with open(splits_path, "r") as f:
        data = json.load(f)
    folds = data["folds"]
    # convert to dict of lists for convenience: idx -> {train,val,test}
    out = {}
    for i, fold in enumerate(folds):
        out[str(i)] = {
            "train": fold["train_files"],
            "val": fold["val_files"],
            "test": fold["test_files"],
            "test_subject": fold["test_subject"],
        }
    return out


# ---------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------
def make_dataloaders(
    split_id: int,
    batch_size: int = 4,
    num_workers: int = 4,
    mode: str = "video",              # 'video' or 'rep'
    augment: bool = False,
    allow_hflip: bool = False,
    num_frames: int = 32,
    resize: int = 224,
    include_pose: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    splits = load_splits()
    key = str(split_id)
    if key not in splits:
        raise KeyError(f"No split '{split_id}' in splits.json")

    train_files = splits[key]["train"]
    val_files   = splits[key]["val"]
    test_files  = splits[key]["test"]

    train_ds = VideoDataset(
        split_files=train_files,
        mode=mode,
        num_frames=num_frames,
        resize=resize,
        augment=augment,
        allow_hflip=allow_hflip,
        include_pose=include_pose,
    )
    val_ds = VideoDataset(
        split_files=val_files,
        mode=mode,
        num_frames=num_frames,
        resize=resize,
        augment=False,                 # never augment val/test
        allow_hflip=False,
        include_pose=include_pose,
    )
    test_ds = VideoDataset(
        split_files=test_files,
        mode=mode,
        num_frames=num_frames,
        resize=resize,
        augment=False,
        allow_hflip=False,
        include_pose=include_pose,
    )

    # simple default collation (fixed-size tensors)
    collate = None

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )

    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )

    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )

    return train_dl, val_dl, test_dl
