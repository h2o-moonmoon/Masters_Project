import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from dataset import KpSeqDataset, EXERCISES, FORM_LABELS
import numpy as np
from sklearn.metrics import classification_report

class CNNBiLSTM(nn.Module):
    def __init__(self, in_dim=34, hidden=128, lstm=128, num_ex=5, num_form=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, 128)
        self.conv = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(128, lstm, bidirectional=True, batch_first=True)
        self.head_ex = nn.Linear(2*lstm, num_ex)
        self.head_form = nn.Linear(2*lstm, num_form)

    def forward(self, x):
        x = self.proj(x)            # (B,T,128)
        x = x.transpose(1,2)        # (B,128,T)
        x = self.relu(self.conv(x)) # (B,128,T)
        x = x.transpose(1,2)        # (B,T,128)
        out, _ = self.lstm(x)       # (B,T,2*lstm)
        h = out.mean(dim=1)         # (B,2*lstm)
        return self.head_ex(h), self.head_form(h)

def train_one_epoch(model, dl, opt, device, class_weights_form=None):
    model.train()
    ce_ex = nn.CrossEntropyLoss()
    ce_form = nn.CrossEntropyLoss(weight=class_weights_form)  # handle imbalance
    total = 0; loss_sum = 0
    for x, y_ex, y_form in dl:
        x, y_ex, y_form = x.to(device), y_ex.to(device), y_form.to(device)
        opt.zero_grad()
        logits_ex, logits_form = model(x)
        loss = ce_ex(logits_ex, y_ex) + ce_form(logits_form, y_form)
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
        le, lf = model(x)
        preds_ex += le.argmax(1).cpu().tolist(); gts_ex += y_ex.tolist()
        preds_fm += lf.argmax(1).cpu().tolist(); gts_fm += y_form.tolist()
    return (preds_ex, gts_ex), (preds_fm, gts_fm)

def compute_form_class_weights(dataset):
    # weights ~ inverse freq
    counts = np.bincount((dataset.df["form"]=="correct").map({False:0, True:1}).values, minlength=2)
    counts[counts==0] = 1
    inv = 1.0 / counts
    w = inv / inv.sum() * 2.0
    return torch.tensor(w, dtype=torch.float32)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = KpSeqDataset("data/annotations/labels.csv", "train")
    val_set   = KpSeqDataset("data/annotations/labels.csv", "val")

    dl_tr = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    dl_va = DataLoader(val_set,   batch_size=16, shuffle=False, num_workers=4)

    model = CNNBiLSTM(in_dim=34, num_ex=len(EXERCISES), num_form=2).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Optional: class weights for form if heavily imbalanced
    class_weights_form = compute_form_class_weights(train_set).to(device)

    for epoch in range(30):
        loss = train_one_epoch(model, dl_tr, opt, device, class_weights_form)
        (pe, ge), (pf, gf) = evaluate(model, dl_va, device)

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} loss {loss:.4f}")
            # Exercise report
            print("Exercise:\n", classification_report(
                ge, pe, labels=list(range(len(EXERCISES))),
                target_names=EXERCISES, zero_division=0))
            # Binary form report (0=incorrect, 1=correct)
            print("Form:\n", classification_report(
                gf, pf, labels=[0,1],
                target_names=FORM_LABELS, zero_division=0))

    torch.save({
        "model": model.state_dict(),
        "exercise_labels": EXERCISES,
        "form_labels": FORM_LABELS
    }, "outputs/models/seq_model.pt")
