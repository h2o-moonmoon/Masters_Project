import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from pathlib import Path
import random

EXERCISES = ["deadlift","lunge","pushup","row","squat"]
FORM_LABELS = ["incorrect","correct"]  # 0, 1

def to_id(value, vocab):
    return vocab.index(value)

class KpSeqDataset(Dataset):
    def __init__(self, labels_csv, split, seq_len=64, stride=2, drop_conf=True, path_col="video_path"):
        self.df = pd.read_csv(labels_csv, sep='\t')
        self.df = self.df[self.df["split"]==split].reset_index(drop=True)

        # Ensure we have a 'form' column as incorrect/correct.
        # If only form_subtype is present, derive form from it.
        if "form" not in self.df.columns:
            if "form_subtype" in self.df.columns:
                self.df["form"] = np.where(self.df["form_subtype"].fillna("none").astype(str).str.lower() == "none",
                                           "correct", "incorrect")
            else:
                raise ValueError("labels CSV must contain 'form' or 'form_subtype' to derive it.")

        # Normalize values
        self.df["exercise"] = self.df["exercise"].str.lower()
        self.df["form"] = self.df["form"].str.lower()

        self.seq_len = seq_len
        self.stride = stride
        self.drop_conf = drop_conf
        self.kp_dir = Path("outputs/keypoints")
        self.path_col = path_col

    def __len__(self):
        return len(self.df)

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
        stem = Path(row[self.path_col]).stem
        kp_path = self.kp_dir / f"{stem}.npy"
        kp = np.load(kp_path)  # (T,17,3)
        kp = self._temporal_slice(kp)
        if self.drop_conf:
            kp = kp[..., :2]

        # Normalize coordinates (adjust divisor if your preprocess chose different size)
        feat = (kp / 640.0).reshape(kp.shape[0], -1).astype(np.float32)  # (T,34)
        x = torch.from_numpy(feat)

        y_ex = to_id(row["exercise"], EXERCISES)
        y_form = 1 if row["form"] == "correct" else 0  # incorrect=0, correct=1

        return x, torch.tensor(y_ex), torch.tensor(y_form)
