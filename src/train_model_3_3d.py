import argparse, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path

from torchvision.models.video import r2plus1d_18, r3d_18, mc3_18
try:
    from torchvision.models.video import R2Plus1D_18_Weights, R3D_18_Weights, MC3_18_Weights
    HAS_WEIGHTS = True
except Exception:
    HAS_WEIGHTS = False

from datasets.core import make_dataloaders, ROOT
from train_model_1 import (
    multitask_loss,
    compute_loss_for_task,
    evaluate,
    set_seed,
    ROOT,
)
from utils.metrics import save_confusion


class Video3D_MTL(nn.Module):
    """
    Wrap a torchvision 3D backbone and attach multi-task heads:
      - exercise (5-way)
      - form (2-way)
      - type (3-way)  [evaluated only when form != 0]
    Expects input x as [B, T, C, H, W]; we permute to [B, C, T, H, W] internally.
    """
    def __init__(self, arch="r2plus1d_18",
                 exercise_classes=5, form_classes=2, type_classes=3, pretrained=True):
        super().__init__()

        # --- build backbone ---
        if arch == "r2plus1d_18":
            if HAS_WEIGHTS and pretrained:
                m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
            else:
                m = r2plus1d_18(weights=None)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()  # output: [B, feat_dim]
        elif arch == "r3d_18":
            if HAS_WEIGHTS and pretrained:
                m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            else:
                m = r3d_18(weights=None)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
        elif arch == "mc3_18":
            if HAS_WEIGHTS and pretrained:
                m = mc3_18(weights=MC3_18_Weights.KINETICS400_V1)
            else:
                m = mc3_18(weights=None)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.backbone = m
        self.feat_dim = int(feat_dim)

        # --- heads ---
        self.head_ex = nn.Linear(self.feat_dim, exercise_classes)
        self.head_form = nn.Linear(self.feat_dim, form_classes)
        self.head_type = nn.Linear(self.feat_dim, type_classes)

    def forward(self, x):  # x: [B, T, C, H, W]
        # permute to [B, C, T, H, W] for torchvision video models
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input [B,T,C,H,W], got shape {tuple(x.shape)}")
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        z = self.backbone(x)  # [B, D]
        return self.head_ex(z), self.head_form(z), self.head_type(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--mode", type=str, default="rep", choices=["rep","video"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)   # 3D is heavier; keep small
    ap.add_argument("--frames", type=int, default=16)      # 16 or 32 are common
    ap.add_argument("--size", type=int, default=160)       # 112/128/160/224; 160 is a good middle
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--arch", type=str, default="r2plus1d_18", choices=["r2plus1d_18","r3d_18","mc3_18"])
    ap.add_argument("--pretrained", action="store_true")   # use Kinetics pretrained weights if available
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--type-loss-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--form-class-weights", type=str, default="", help="e.g., '1.0,1.5' for [correct,incorrect]")
    ap.add_argument("--form-focal-gamma", type=float, default=0.0, help=">0 enables focal loss for form")
    ap.add_argument("--form-loss-weight", type=float, default=1.0)
    ap.add_argument(
        "--task", type=str, default="ex_form_type",
        choices=["exercise", "ex_form", "ex_form_type", "per_exercise"],
        help="What to train: exercise only, exercise+form, full multitask, or per-exercise form+type."
    )
    ap.add_argument("--exercise-filter", type=str, default="", help="If set, restrict to one exercise.")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    aug_tag = "aug" if args.augment else "noaug"
    exercise_filter = args.exercise_filter or None
    exercise_tag = exercise_filter if exercise_filter is not None else "all"

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataloaders: same dataset, just different model
    train_dl, val_dl, test_dl = make_dataloaders(
        split_id=args.split,
        batch_size=args.batch_size,
        num_workers=4,
        mode=args.mode,
        augment=args.augment,
        allow_hflip=False,   # keep off for form symmetry concerns
        num_frames=args.frames,
        resize=args.size,
        include_pose=False,
        exercise_filter=exercise_filter,
    )

    model = Video3D_MTL(arch=args.arch, pretrained=args.pretrained).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = float("inf")
    model_tag = f"video3d_{args.arch}_{args.task}_{exercise_tag}_{aug_tag}"
    outdir = ROOT / "analysis" / model_tag / f"split{args.split}_{args.mode}"
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{model_tag}_split{args.split}_{args.mode}_seed{args.seed}.pt"

    # training loop
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_dl:
            x = batch["video"].to(device)       # [B,T,C,H,W]
            y_ex = batch["exercise_id"].to(device)
            y_form = batch["form_id"].to(device)
            y_type = batch["subtype_id"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                l_ex, l_form, l_type = model(x)
                loss, parts = compute_loss_for_task(
                    l_ex, l_form, l_type,
                    y_ex, y_form, y_type,
                    args
                )

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            total += loss.item() * x.size(0)

        # eval (reuse your evaluate; it expects model(x) -> 3 logits and batch["video"])
        val = evaluate(model, val_dl, device, args, input_key="video")
        test = evaluate(model, test_dl, device, args, input_key="video")

        print(f"[{ep:03d}] train_loss={total/len(train_dl.dataset):.4f} | "
              f"val_loss={val['loss']:.4f} val_exF1={val['exercise_f1']:.3f} val_formF1={val['form_f1']:.3f} val_typeF1={val['type_f1']:.3f} | "
              f"test_exF1={test['exercise_f1']:.3f} test_formF1={test['form_f1']:.3f} test_typeF1={test['type_f1']:.3f}")

        # save best-by-val-loss
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save({"model": model.state_dict(), "args": vars(args), "val": val}, ckpt_path)

        # confusion matrices per epoch
        form_names = ["correct","incorrect"]
        type_names = ["upper","lower"]
        save_confusion(val["form_true"],  val["form_pred"],  form_names, outdir / f"val_form_cm_epoch{ep:03d}")
        save_confusion(val["type_true"],  val["type_pred"],  type_names, outdir / f"val_type_cm_epoch{ep:03d}")
        save_confusion(test["form_true"], test["form_pred"], form_names, outdir / f"test_form_cm_epoch{ep:03d}")
        save_confusion(test["type_true"], test["type_pred"], type_names, outdir / f"test_type_cm_epoch{ep:03d}")

    print(f"Best checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
