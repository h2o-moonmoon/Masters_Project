import re, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
OUT_CSV = ROOT / "analysis" / "summary_runs.csv"
OUT_CSV.parent.mkdir(exist_ok=True)

# Regex to capture your printed line fields (adjust if your prints differ)
line_re = re.compile(
    r"\[(\d+)\]\s+train_loss=([0-9.]+)\s+\|\s+val_loss=([0-9.]+).*val_exF1=([0-9.]+)\s+val_formF1=([0-9.]+)\s+val_typeF1=([0-9.]+)\s+\|\s+test_exF1=([0-9.]+)\s+test_formF1=([0-9.]+)\s+test_typeF1=([0-9.]+)"
)

def parse_one_log(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = line_re.search(line)
            if m:
                ep = int(m.group(1))
                rows.append({
                    "epoch": ep,
                    "train_loss": float(m.group(2)),
                    "val_loss": float(m.group(3)),
                    "val_exF1": float(m.group(4)),
                    "val_formF1": float(m.group(5)),
                    "val_typeF1": float(m.group(6)),
                    "test_exF1": float(m.group(7)),
                    "test_formF1": float(m.group(8)),
                    "test_typeF1": float(m.group(9)),
                })
    return rows

def main():
    out = []
    for lf in LOG_DIR.glob("*.log"):
        # parse model/split/mode/seed from filename
        name = lf.stem  # e.g. rgb_attnpool_resnet18_split0_rep_seed42
        parts = name.split("_")
        # crude parse:
        model_tag = "_".join(parts[:-4])  # everything up to split
        split = parts[-4].replace("split","")
        mode  = parts[-3]
        seed  = parts[-1].replace("seed","")

        rows = parse_one_log(lf)
        if not rows:
            continue
        best = min(rows, key=lambda r: r["val_loss"])
        out.append({
            "model": model_tag, "split": split, "mode": mode, "seed": seed, **best
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"Wrote {OUT_CSV} with {len(out)} rows")

if __name__ == "__main__":
    main()
