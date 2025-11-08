import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from dataset import KpSeqDataset, EXERCISES
import pandas as pd, numpy as np
from sklearn.metrics import classification_report

class CNNBiLSTM(nn.Module):
    def __init__(self, in_dim=34, hidden=128, lstm=128, num_ex=5, num_form=8):
        super().__init__()
        # temporal conv over (T, in_dim) -> treat in_dim as channels with 1D conv across time
        self.proj = nn.Linear(in_dim, 128)
        self.conv = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(128, lstm, bidirectional=True, batch_first=True)
        self.head_ex = nn.Linear(2*lstm, num_ex)
        self.head_form = nn.Linear(2*lstm, num_form)

    def forward(self, x):
        # x: (B, T, F)
        x = self.proj(x)                   # (B,T,128)
        x = x.transpose(1,2)               # (B,128,T)
        x = self.relu(self.conv(x))        # (B,128,T)
        x = x.transpose(1,2)               # (B,T,128)
        out, _ = self.lstm(x)              # (B,T,2*lstm)
        h = out.mean(dim=1)                # temporal average
        return self.head_ex(h), self.head_form(h)

def train_one_epoch(model, dl, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0; loss_sum = 0
    for x, y_ex, y_form in dl:
        x, y_ex, y_form = x.to(device), y_ex.to(device), y_form.to(device)
        opt.zero_grad()
        ex, form = model(x)
        loss = ce(ex, y_ex) + ce(form, y_form)
        loss.backward(); opt.step()
        loss_sum += loss.item()*x.size(0); total += x.size(0)
    return loss_sum/total

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    preds_ex, gts_ex = [], []
    preds_fm, gts_fm = [], []
    for x, y_ex, y_form in dl:
        x = x.to(device)
        ex, fm = model(x)
        preds_ex += ex.argmax(1).cpu().tolist(); gts_ex += y_ex.tolist()
        preds_fm += fm.argmax(1).cpu().tolist(); gts_fm += y_form.tolist()
    return (preds_ex, gts_ex), (preds_fm, gts_fm)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = KpSeqDataset("data/annotations/labels.csv", "train")
    val_set   = KpSeqDataset("data/annotations/labels.csv", "val")
    num_form = len(train_set.form_vocab)

    dl_tr = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    dl_va = DataLoader(val_set,   batch_size=16, shuffle=False, num_workers=4)

    model = CNNBiLSTM(in_dim=34, num_ex=len(EXERCISES), num_form=num_form).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    for epoch in range(30):
        loss = train_one_epoch(model, dl_tr, opt, device)
        (pe, ge), (pf, gf) = evaluate(model, dl_va, device)
        if (epoch+1)%5==0:
            print(f"Epoch {epoch+1} loss {loss:.4f}")
            print("Exercise:\n", classification_report(ge, pe, target_names=EXERCISES))
            # map form IDs back to names:
            print("Form:\n", classification_report(gf, pf, target_names=train_set.form_vocab))
    torch.save({"model": model.state_dict(),
                "form_vocab": train_set.form_vocab}, "outputs/models/seq_model.pt")
