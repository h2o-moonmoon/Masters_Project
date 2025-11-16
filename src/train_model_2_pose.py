import argparse, torch, torch.optim as optim
from torch.utils.data import DataLoader
from datasets.core import load_splits
from datasets.pose import PoseDataset
from models.pose_rnn import PoseGRU_MTL
from train_model_1 import multitask_loss, evaluate, set_seed, ROOT
from utils.metrics import save_confusion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--mode", type=str, default="rep", choices=["rep","video"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--type-loss-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--form-class-weights", type=str, default="", help="e.g., '1.0,1.5' for [correct,incorrect]")
    ap.add_argument("--form-focal-gamma", type=float, default=0.0, help=">0 enables focal loss for form")
    ap.add_argument("--form-loss-weight", type=float, default=1.0)
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    aug_tag = "aug" if args.augment else "noaug"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits()
    key = str(args.split)
    train_ds = PoseDataset(splits[key]["train"], mode=args.mode)
    val_ds   = PoseDataset(splits[key]["val"],   mode=args.mode)
    test_ds  = PoseDataset(splits[key]["test"],  mode=args.mode)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = PoseGRU_MTL(H=128).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    best = 1e9
    for ep in range(1, args.epochs+1):
        model.train()
        total = 0.0
        for b in train_dl:
            opt.zero_grad(set_to_none=True)
            y_ex, y_form, y_type = b["exercise_id"].to(device), b["form_id"].to(device), b["subtype_id"].to(device)
            l_ex, l_form, l_type = model(b["pose"].to(device))
            loss, _ = multitask_loss(l_ex, l_form, l_type, y_ex, y_form, y_type, args.type_loss_weight)
            loss.backward(); opt.step()
            total += loss.item() * y_ex.size(0)
        val = evaluate(model, val_dl, device, args, input_key="pose")
        test = evaluate(model, test_dl, device, args, input_key="pose")
        print(f"[{ep:03d}] train_loss={total/len(train_ds):.4f} | "
              f"val_formF1={val['form_f1']:.3f} val_typeF1={val['type_f1']:.3f} | "
              f"test_formF1={test['form_f1']:.3f} test_typeF1={test['type_f1']:.3f}")
        model_tag = f"pose_rnn_gru128_{aug_tag}"  # adjust per hidden size, layers, etc.
        outdir = ROOT / "analysis" / model_tag / f"split{args.split}_{args.mode}"
        outdir.mkdir(parents=True, exist_ok=True)

        form_names = ["correct", "incorrect"]
        type_names = ["upper", "lower"]

        # VAL
        save_confusion(val["form_true"], val["form_pred"],
                       form_names, outdir / f"val_form_cm_epoch{ep:03d}", title="Val Form")
        save_confusion(val["type_true"], val["type_pred"],
                       type_names, outdir / f"val_type_cm_epoch{ep:03d}", title="Val Type (incorrect only)")

        # TEST
        save_confusion(test["form_true"], test["form_pred"],
                       form_names, outdir / f"test_form_cm_epoch{ep:03d}", title="Test Form")
        save_confusion(test["type_true"], test["type_pred"],
                       type_names, outdir / f"test_type_cm_epoch{ep:03d}", title="Test Type (incorrect only)")


if __name__ == "__main__":
    main()
