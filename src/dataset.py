import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from pathlib import Path
import random

EXERCISES = ["deadlift","lunge","pushup","row","squat"]

def to_id(value, vocab):
    return vocab.index(value)

class KpSeqDataset(Dataset):
    def __init__(self, labels_csv, split, seq_len=64, stride=2, drop_conf=True):
        self.df = pd.read_csv(labels_csv, sep='\t')
        self.df = self.df[self.df["split"]==split].reset_index(drop=True)
        self.seq_len = seq_len
        self.stride = stride
        self.drop_conf = drop_conf
        self.kp_dir = Path("outputs/keypoints")

        # build form vocab dynamically (correct + your three incorrects per exercise)
        self.form_vocab = sorted(self.df["form_subtype"].fillna("none").unique().tolist())

    def __len__(self): return len(self.df)

    def _temporal_slice(self, seq):
        T = seq.shape[0]
        if T <= self.seq_len:
            pad = np.repeat(seq[-1:], self.seq_len-T, axis=0) if T>0 else np.zeros((self.seq_len, seq.shape[1], seq.shape[2]))
            return np.concatenate([seq, pad], axis=0)
        start = random.randint(0, max(0, T - self.seq_len*self.stride))
        idxs = np.arange(start, start+self.seq_len*self.stride, self.stride)
        idxs = np.clip(idxs, 0, T-1)
        return seq[idxs]

    def __getitem__(self, i):
        row = self.df.iloc[i]
        kp = np.load(self.kp_dir / (Path(row["video_path"]).stem + ".npy"))  # (T,17,3)
        kp = self._temporal_slice(kp)
        if self.drop_conf: kp = kp[..., :2]
        # normalize by image size: assume coords already in pixels; divide by max(w,h) if you saved it
        # here assume coords in resized frame; scale to [0,1] by dividing by 640 (adjust if needed)
        kp = kp / 640.0

        # shape to (C,T) for simple CNN1D over joints
        # Option 1: flatten joints: (T, 17*2)
        feat = kp.reshape(kp.shape[0], -1).astype(np.float32)  # (T, 34)
        x = torch.from_numpy(feat)                             # (T, 34)

        y_ex = to_id(row["exercise"], EXERCISES)
        y_form = to_id(row["form_subtype"] if isinstance(row["form_subtype"], str) else "none", self.form_vocab)
        return x, torch.tensor(y_ex), torch.tensor(y_form)
