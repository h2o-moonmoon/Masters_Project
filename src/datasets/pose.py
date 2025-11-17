from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import json

from datasets.core import ROOT, META_CSV, SEG_JSON, build_label_maps, POSES_DIR

class PoseDataset(Dataset):
    def __init__(self, split_files, metadata_csv=META_CSV, segments_json=SEG_JSON,
                 mode="rep", num_frames=48, exercise_filter=None):
        self.mode = mode
        self.num_frames = num_frames
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df["file"].isin(split_files)].copy()
        self.exercise_filter = exercise_filter
        if self.exercise_filter is not None:
            self.df = self.df[self.df["exercise"] == self.exercise_filter].copy()
            if len(self.df) == 0:
                raise ValueError(f"No samples for exercise_filter='{self.exercise_filter}'")
        self.maps = build_label_maps(self.df)
        with open(segments_json, "r") as f:
            self.segs = json.load(f)
        self.samples = []
        for _, r in self.df.iterrows():
            stem = Path(r["file"]).stem
            base = dict(stem=stem,
                        ex=self.maps["exercise2id"][r["exercise"]],
                        form=self.maps["form2id"][r["form"]],
                        subtype=int(r["subtype"]))
            if self.mode == "video":
                self.samples.append({**base, "segment": None})
            else:
                seg = self.segs.get(stem, {"segments": None})
                if seg["segments"] is None:
                    for k in range(3):
                        self.samples.append({**base, "segment": None})
                else:
                    for (s,e) in seg["segments"]:
                        self.samples.append({**base, "segment": (int(s),int(e))})

    def __len__(self): return len(self.samples)

    def _load_pose(self, stem):
        npz = np.load(POSES_DIR / f"{stem}.npz")
        kpts = npz["kpts"].astype(np.float32)  # [T,J,2]
        conf = npz["confs"].astype(np.float32) # [T,J]
        # center + scale
        L_SHO,R_SHO,L_HIP,R_HIP,PELVIS = 5,6,11,12,11
        center = kpts[:, PELVIS]
        if np.isnan(center).any():
            center = np.nanmean(kpts[:, [L_HIP, R_HIP]], axis=1)
        kp = kpts - center[:,None,:]
        sh = np.linalg.norm(kpts[:,L_SHO]-kpts[:,R_SHO], axis=1)
        hip = np.linalg.norm(kpts[:,L_HIP]-kpts[:,R_HIP], axis=1)
        scale = np.nanmedian((sh+hip)/2.0);  scale = 1.0 if not np.isfinite(scale) or scale==0 else scale
        kp = kp / scale
        # replace NaNs with 0, keep a mask if you want later
        kp = np.nan_to_num(kp, nan=0.0)
        return kp, conf

    def _sample_indices(self, T, segment):
        if segment is None:
            idx = np.linspace(0, T-1, num=min(self.num_frames,T), dtype=np.int32)
        else:
            s,e = segment
            e = min(e, T-1); s = max(0, s)
            L = max(1, e-s+1)
            rel = np.linspace(0, L-1, num=min(self.num_frames,L), dtype=np.int32)
            idx = s + rel
        if len(idx) < self.num_frames:
            idx = np.pad(idx, (0, self.num_frames-len(idx)), mode="edge")
        return idx

    def __getitem__(self, i):
        it = self.samples[i]
        kp, conf = self._load_pose(it["stem"])
        T = kp.shape[0]
        idx = self._sample_indices(T, it["segment"])
        seq = kp[idx]                      # [T', J, 2]
        seq = torch.from_numpy(seq).float()
        return {
            "pose": seq,                   # [T, J, 2]
            "exercise_id": torch.tensor(it["ex"]).long(),
            "form_id": torch.tensor(it["form"]).long(),
            "subtype_id": torch.tensor(it["subtype"]).long(),
            "file_stem": it["stem"],
        }
