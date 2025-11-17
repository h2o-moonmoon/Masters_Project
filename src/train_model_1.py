import os, math, json, random, argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from datasets.core import make_dataloaders, ROOT

import matplotlib.pyplot as plt
from utils.metrics import save_confusion


# -------------------------
# Model: 2D -> Temporal Attention -> Multi-task heads
# -------------------------
class TemporalAttention(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        # use a small MLP for attention scores
        mid = max(32, d // 2)
        self.score = nn.Sequential(
            nn.Linear(d, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, 1),
        )

    def forward(self, x):  # x: [B, T, D]
        s = self.score(x)              # [B, T, 1]
        w = torch.softmax(s, dim=1)    # attention weights over time
        z = (w * x).sum(dim=1)         # [B, D]
        return z, w


class VideoAttnPoolMTL(nn.Module):
    def __init__(self, exercise_classes=5, form_classes=2, type_classes=3, backbone='resnet18'):
        super().__init__()
        # --- build a 2D backbone (global pooled feature per frame) ---
        from torchvision.models import resnet18, resnet50
        try:
            # Newer torchvision API
            if backbone == 'resnet18':
                m = resnet18(weights="IMAGENET1K_V1")
            else:
                m = resnet50(weights="IMAGENET1K_V1")
        except Exception:
            # Fallback for older torchvision
            if backbone == 'resnet18':
                m = resnet18(weights=None)  # will still work with random init
            else:
                m = resnet50(weights=None)

        # keep everything up to the avgpool (remove final fc)
        self.backbone = nn.Sequential(*(list(m.children())[:-1]))  # -> [B, D, 1, 1]

        # --- infer feature dimension robustly ---
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.zeros(1, 3, 224, 224)     # do not .to(device) here; just infer shape
            f = self.backbone(dummy).flatten(1)     # [1, D]
            feat_dim = f.shape[1]

        self.feat_dim = int(feat_dim)

        # --- temporal attention & heads built with the inferred D ---
        self.temporal = TemporalAttention(self.feat_dim)
        self.head_ex = nn.Linear(self.feat_dim, exercise_classes)
        self.head_form = nn.Linear(self.feat_dim, form_classes)
        self.head_type = nn.Linear(self.feat_dim, type_classes)

    def frame_encode(self, frames):  # frames: [B*T, 3, H, W]
        f = self.backbone(frames)    # [B*T, D, 1, 1]
        return f.flatten(1)          # [B*T, D]

    def forward(self, x):            # x: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        f = self.frame_encode(x)                 # [B*T, D]
        f = f.view(B, T, self.feat_dim)          # [B, T, D]
        z, att = self.temporal(f)                # [B, D], [B, T, 1]
        assert f.shape[-1] == self.feat_dim, f"Backbone feature dim {f.shape[-1]} != expected {self.feat_dim}"
        return self.head_ex(z), self.head_form(z), self.head_type(z), att



# -------------------------
# Loss (with masking for type when form==correct)
# -------------------------
def multitask_loss(logits_ex, logits_form, logits_type, y_ex, y_form, y_type, type_mask_weight=1.0):
    ce = nn.CrossEntropyLoss()
    loss_ex = ce(logits_ex, y_ex)
    loss_form = ce(logits_form, y_form)

    # mask type loss when form is "correct" (assume form_id 0 == 'correct')
    incorrect_mask = (y_form != 0).float()  # 1 for incorrect items
    if incorrect_mask.sum() > 0:
        # compute CE only on incorrect subset
        ce_type = nn.CrossEntropyLoss(reduction='none')(logits_type, y_type)
        loss_type = (ce_type * incorrect_mask).sum() / (incorrect_mask.sum())
    else:
        loss_type = torch.tensor(0.0, device=logits_type.device)

    total = loss_ex + loss_form + type_mask_weight * loss_type
    return total, {"ex": loss_ex.item(), "form": loss_form.item(), "type": loss_type.item()}

def compute_loss_for_task(logits_ex, logits_form, logits_type,
                          y_ex, y_form, y_type, args):
    """
    Wraps multitask_loss but can drop some heads depending on args.task.
    """
    task = getattr(args, "task", "ex_form_type")

    if task == "exercise":
        # exercise-only model: ignore form and type entirely
        ce_ex = torch.nn.CrossEntropyLoss()
        loss_ex = ce_ex(logits_ex, y_ex)
        return loss_ex, {"ex": loss_ex.item(), "form": 0.0, "type": 0.0}

    elif task == "ex_form":
        # exercise + form: no type loss
        ce_ex = torch.nn.CrossEntropyLoss()
        ce_form = torch.nn.CrossEntropyLoss()
        loss_ex = ce_ex(logits_ex, y_ex)
        loss_form = ce_form(logits_form, y_form)
        total = loss_ex + loss_form
        return total, {"ex": loss_ex.item(), "form": loss_form.item(), "type": 0.0}

    else:
        # full model (exercise + form + type) — existing behaviour
        total, parts = multitask_loss(
            logits_ex, logits_form, logits_type,
            y_ex, y_form, y_type,
            type_mask_weight=args.type_loss_weight
        )
        return total, parts


# -------------------------
# Metrics
# -------------------------
def compute_metrics(y_true, y_pred, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return acc, f1


# -------------------------
# Training Loop
# -------------------------
def train_one_epoch(model, loader, optim, device, scaler, args):
    model.train()
    total_loss = 0.0
    for batch in loader:
        vid = batch["video"].to(device)
        y_ex = batch["exercise_id"].to(device)
        y_form = batch["form_id"].to(device)
        y_type = batch["subtype_id"].to(device)

        optim.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            l_ex, l_form, l_type, _ = model(vid)
            loss, parts = compute_loss_for_task(
                l_ex, l_form, l_type,
                y_ex, y_form, y_type,
                args
            )

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        total_loss += loss.item() * vid.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, args, input_key: str = "video"):
    task = getattr(args, "task", "ex_form_type")
    model.eval()
    losses = []
    ex_true, ex_pred = [], []
    form_true, form_pred = [], []
    type_true, type_pred, type_mask = [], [], []

    for batch in loader:
        x = batch[input_key].to(device)                    # <<-- NEW: "video" or "pose"
        y_ex = batch["exercise_id"].to(device)
        y_form = batch["form_id"].to(device)
        y_type = batch["subtype_id"].to(device)

        # Support models that return (ex, form, type, att) OR just (ex, form, type)
        outs = model(x)
        if isinstance(outs, (list, tuple)):
            if len(outs) >= 3:
                l_ex, l_form, l_type = outs[:3]
            else:
                raise RuntimeError(f"Model returned {len(outs)} outputs, expected ≥3.")
        else:
            raise RuntimeError("Model should return a tuple/list of logits.")

        # loss, _ = multitask_loss(l_ex, l_form, l_type, y_ex, y_form, y_type, args.type_loss_weight)
        loss, _ = compute_loss_for_task(
            l_ex, l_form, l_type,
            y_ex, y_form, y_type,
            args
        )
        losses.append(loss.item() * x.size(0))

        ex_true.extend(y_ex.cpu().tolist());     ex_pred.extend(l_ex.argmax(1).cpu().tolist())
        form_true.extend(y_form.cpu().tolist()); form_pred.extend(l_form.argmax(1).cpu().tolist())

        inc_mask = (y_form != 0)
        if inc_mask.any():
            type_true.extend(y_type[inc_mask].cpu().tolist())
            type_pred.extend(l_type[inc_mask].argmax(1).cpu().tolist())
            type_mask.extend([1] * int(inc_mask.sum().item()))

    # Always compute exercise metrics (even if not trained on them, for inspection)
    ex_acc, ex_f1 = compute_metrics(ex_true, ex_pred, average='macro')

    # Form metrics only if model is supervising form
    if task in ("ex_form", "ex_form_type"):
        form_acc, form_f1 = compute_metrics(form_true, form_pred, average='macro')
    else:
        form_acc, form_f1 = float('nan'), float('nan')

    # Type metrics only if model is supervising type
    if task == "ex_form_type" and len(type_mask) > 0:
        type_acc, type_f1 = compute_metrics(type_true, type_pred, average='macro')
    else:
        type_acc, type_f1 = float('nan'), float('nan')

    return {
        "loss": sum(losses) / len(loader.dataset),
        "exercise_acc": ex_acc, "exercise_f1": ex_f1,
        "form_acc": form_acc, "form_f1": form_f1,
        "type_acc": type_acc, "type_f1": type_f1,
        "ex_true": ex_true, "ex_pred": ex_pred,
        "form_true": form_true, "form_pred": form_pred,
        "type_true": type_true, "type_pred": type_pred
    }

def focal_ce_form(logits, target, gamma=2.0, alpha=None):
    # logits: [B,2], target: [B]
    ce = nn.CrossEntropyLoss(reduction='none', weight=alpha)(logits, target)
    with torch.no_grad():
        pt = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1).clamp_(1e-6, 1.0)
    return ((1-pt)**gamma * ce).mean()


def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=0, help="LOSO fold id (0..2)")
    ap.add_argument("--mode", type=str, default="rep", choices=["rep","video"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","resnet50"])
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--type-loss-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--form-class-weights", type=str, default="", help="e.g., '1.0,1.5' for [correct,incorrect]")
    ap.add_argument("--form-focal-gamma", type=float, default=0.0, help=">0 enables focal loss for form")
    ap.add_argument("--form-loss-weight", type=float, default=1.0)
    ap.add_argument(
        "--task", type=str, default="ex_form_type",
                    choices=["exercise", "ex_form", "ex_form_type"],
                    help="What to train: exercise only, exercise+form, or exercise+form+type."
    )
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    aug_tag = "aug" if args.augment else "noaug"
    task_tag = args.task

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders (always run from project root!)
    train_dl, val_dl, test_dl = make_dataloaders(
        split_id=args.split,
        batch_size=args.batch_size,
        num_workers=4,
        mode=args.mode,
        augment=args.augment,
        allow_hflip=False,         # keep False if left/right matters
        num_frames=args.frames,
        resize=args.size,
        include_pose=False,
    )

    # Model & optimizer
    model = VideoAttnPoolMTL(backbone=args.backbone).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler(enabled=args.amp)

    best_val = float("inf")
    ckpt = ROOT / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    ckpt_path = ckpt / f"attnpool_{task_tag}_{aug_tag}_split{args.split}_{args.mode}_seed{args.seed}.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_dl, opt, device, scaler, args)
        val_metrics = evaluate(model, val_dl, device, args)
        test_metrics = evaluate(model, test_dl, device, args)

        # choose a unique tag for this model
        model_tag = f"rgb_attnpool_{args.backbone}_{task_tag}_{aug_tag}"
        outdir = ROOT / "analysis" / model_tag / f"split{args.split}_{args.mode}"
        outdir.mkdir(parents=True, exist_ok=True)

        # names
        form_names = ["correct", "incorrect"]
        type_names = ["upper", "lower"]  # type is evaluated only on incorrect items

        # VAL
        save_confusion(val_metrics["form_true"], val_metrics["form_pred"],
                       form_names, outdir / f"val_form_cm_epoch{epoch:03d}", title="Val Form")
        save_confusion(val_metrics["type_true"], val_metrics["type_pred"],
                       type_names, outdir / f"val_type_cm_epoch{epoch:03d}", title="Val Type (incorrect only)")

        # TEST
        save_confusion(test_metrics["form_true"], test_metrics["form_pred"],
                       form_names, outdir / f"test_form_cm_epoch{epoch:03d}", title="Test Form")
        save_confusion(test_metrics["type_true"], test_metrics["type_pred"],
                       type_names, outdir / f"test_type_cm_epoch{epoch:03d}", title="Test Type (incorrect only)")

        print(f"[{epoch:03d}] train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} val_exF1={val_metrics['exercise_f1']:.3f} val_formF1={val_metrics['form_f1']:.3f} val_typeF1={val_metrics['type_f1']:.3f} | "
              f"test_exF1={test_metrics['exercise_f1']:.3f} test_formF1={test_metrics['form_f1']:.3f} test_typeF1={test_metrics['type_f1']:.3f}")

        # save best by val loss
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "val": val_metrics
            }, ckpt_path)

    print(f"Best checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
