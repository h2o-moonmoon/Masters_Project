import subprocess, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "python"  # or full path to python if needed

EXPS = [
    # (name, script, fixed_args_dict)
    ("rgb_attnpool_resnet18", "src/train_model_1.py", {"backbone":"resnet18"}),
    ("pose_rnn_gru128",       "src/train_model_2_pose.py", {}),
    ("video3d_r2plus1d18",    "src/train_model_3_3d.py", {"arch":"r2plus1d_18","pretrained":True}),
]

FOLDS = [0,1,2]
MODES = ["rep"]  # add "video" later if you want
AUGS  = [True, False]
SEEDS = [42]     # expand to [42,123,999] for full runs
EPOCHS = 10      # set 30–50 for full runs
FRAMES = 16      # RGB/Video3D; bump to 32 later if GPU allows
SIZE   = 224     # 160 for 3D default in its trainer; we pass per script args

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

def build_cmd(script, args):
    cmd = [PY, script]
    for k, v in args.items():
        if isinstance(v, bool):
            if v: cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd

def exists_checkpoint(name, split, mode, seed):
    ck_dir = ROOT / "checkpoints"
    # match the patterns used in each script
    if name.startswith("rgb_attnpool"):
        p = ck_dir / f"attnpool_split{split}_{mode}_seed{seed}.pt"
    elif name.startswith("pose_rnn"):
        # pose script didn’t save ckpt; you can add it if you wish; for now always run
        return False
    elif name.startswith("video3d"):
        p = ck_dir / f"{name}_split{split}_{mode}_seed{seed}.pt"
    else:
        return False
    return p.exists()

def main():
    jobs = []
    for name, script, fixed in EXPS:
        for split in FOLDS:
            for mode in MODES:
                for aug in AUGS:
                    for seed in SEEDS:
                        args = dict(split=split, mode=mode, epochs=EPOCHS, seed=seed)
                        if "rgb_attnpool" in name:
                            args.update(dict(frames=FRAMES, size=SIZE))
                            if aug: args["augment"] = True
                            args.update(fixed)
                        elif "pose_rnn" in name:
                            if aug: pass  # Pose aug usually done as pose jitter; skip here
                            args.update(fixed)
                        elif "video3d" in name:
                            # use smaller size default in that trainer if not provided
                            args.update(dict(frames=16, size=160))
                            if aug: args["augment"] = True
                            args.update(fixed)

                        # skip if checkpoint already exists (resumable behavior)
                        if exists_checkpoint(name, split, mode, seed):
                            print(f"[SKIP] {name} split{split} {mode} seed{seed} (ckpt exists)")
                            continue

                        jobs.append((name, script, args))

    print(f"Planned jobs: {len(jobs)}")
    for (name, script, args) in jobs:
        aug_tag = "aug" if args.get("augment") else "noaug"
        tag = f"{name}_{aug_tag}_split{args['split']}_{args['mode']}_seed{args['seed']}"
        log_file = LOG_DIR / f"{tag}.log"
        cmd = build_cmd(script, args)
        print("[RUN]", " ".join(map(str, cmd)))
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("CMD: " + " ".join(map(str, cmd)) + "\n\n")
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            ret = proc.wait()
        if ret != 0:
            print(f"[ERR] Job failed: {tag} (see {log_file})")
        else:
            print(f"[OK]  {tag}  (log: {log_file})")
        time.sleep(1)

if __name__ == "__main__":
    main()
