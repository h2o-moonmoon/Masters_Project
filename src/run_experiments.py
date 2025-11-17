#!/usr/bin/env python3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "python"  # or absolute python path if you prefer

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# --- models and their scripts ---
EXPS = [
    # (name, script, fixed_args_dict)
    ("rgb_attnpool_resnet18", "src/train_model_1.py", {"backbone": "resnet18"}),
    ("pose_rnn_gru128",       "src/train_model_2_pose.py", {}),
    ("video3d_r2plus1d18",    "src/train_model_3_3d.py", {"arch": "r2plus1d_18", "pretrained": True}),
]

# Folds, modes, seeds
FOLDS = [0, 1, 2]
MODES = ["rep"]          # you can add "video" later
SEEDS = [42]

# Training length
EPOCHS_PILOT = 10        # short runs (pilot)
EPOCHS_PEREX = 15        # per-exercise specialist

# Common clip/frame settings
FRAMES_RGB = 16
SIZE_RGB = 224
FRAMES_3D = 16
SIZE_3D = 160

# Task configs
TASKS_GENERAL = ["exercise", "ex_form", "ex_form_type"]
TASK_PER_EXERCISE = "per_exercise"

# Name of exercises as they appear in metadata["exercise"]
EXERCISES = ["Deadlift", "Lunge", "Pushup", "Row", "squat"]

def build_cmd(script, args):
    cmd = [PY, script]
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        elif v is not None:
            cmd += [f"--{k}", str(v)]
    return cmd

def find_ckpt(name, args):
    """Infer a checkpoint path that should be produced by this run.
       If the file exists, we skip the job (resume behaviour).
    """
    ckpt_dir = ROOT / "checkpoints"
    if "rgb_attnpool" in name:
        aug_tag = "aug" if args.get("augment") else "noaug"
        task_tag = args.get("task", "ex_form_type")
        ex_tag = args.get("exercise-filter") or "all"
        fname = f"attnpool_{task_tag}_{ex_tag}_{aug_tag}_split{args['split']}_{args['mode']}_seed{args['seed']}.pt"
        return ckpt_dir / fname
    elif "pose_rnn" in name:
        task_tag = args.get("task", "ex_form_type")
        ex_tag = args.get("exercise-filter") or "all"
        aug_tag = "aug" if args.get("augment") else "noaug"
        fname = f"pose_rnn_{task_tag}_{ex_tag}_{aug_tag}_split{args['split']}_{args['mode']}_seed{args['seed']}.pt"
        return ckpt_dir / fname
    elif "video3d" in name:
        aug_tag = "aug" if args.get("augment") else "noaug"
        task_tag = args.get("task", "ex_form_type")
        ex_tag = args.get("exercise-filter") or "all"
        base = f"video3d_{args.get('arch', 'r2plus1d_18')}_{task_tag}_{ex_tag}_{aug_tag}"
        fname = f"{base}_split{args['split']}_{args['mode']}_seed{args['seed']}.pt"
        return ckpt_dir / fname
    else:
        return None

def main():
    DRY_RUN = "--dry-run" in sys.argv

    jobs = []

    # ========== 1) Generalist models: exercise / ex_form / ex_form_type ==========
    for name, script, fixed in EXPS:
        for split in FOLDS:
            for mode in MODES:
                for seed in SEEDS:
                    for task in TASKS_GENERAL:
                        # Augmentation: for Pose-RNN we might skip augment entirely for now
                        for augment in [False, True]:
                            if "pose_rnn" in name and augment:
                                continue  # Pose aug not implemented; skip

                            args = {
                                "split": split,
                                "mode": mode,
                                "epochs": EPOCHS_PILOT,
                                "seed": seed,
                                "task": task,
                            }

                            # Common settings per model family
                            if "rgb_attnpool" in name:
                                args.update(dict(frames=FRAMES_RGB, size=SIZE_RGB))
                                if augment:
                                    args["augment"] = True
                            elif "pose_rnn" in name:
                                # no video-specific args
                                pass
                            elif "video3d" in name:
                                args.update(dict(frames=FRAMES_3D, size=SIZE_3D))
                                if augment:
                                    args["augment"] = True

                            args.update(fixed)

                            ckpt = find_ckpt(name, args)
                            if ckpt is not None and ckpt.exists():
                                print(f"[SKIP] {name} task={task} split{split} {mode} seed{seed} "
                                      f"augment={augment} (ckpt exists: {ckpt.name})")
                                continue

                            jobs.append((name, script, args))

    # ========== 2) Per-exercise models ==========
    for name, script, fixed in EXPS:
        for split in FOLDS:
            for mode in MODES:
                for seed in SEEDS:
                    for ex_name in EXERCISES:
                        task = TASK_PER_EXERCISE
                        # Aug: you can choose to run only aug or only no-aug for per-exercise
                        for augment in [False, True]:
                            if "pose_rnn" in name and augment:
                                continue  # skip pose aug unless you implement it

                            args = {
                                "split": split,
                                "mode": mode,
                                "epochs": EPOCHS_PEREX,
                                "seed": seed,
                                "task": task,
                                "exercise-filter": ex_name,
                            }

                            if "rgb_attnpool" in name:
                                args.update(dict(frames=FRAMES_RGB, size=SIZE_RGB))
                                if augment:
                                    args["augment"] = True
                            elif "pose_rnn" in name:
                                pass
                            elif "video3d" in name:
                                args.update(dict(frames=FRAMES_3D, size=SIZE_3D))
                                if augment:
                                    args["augment"] = True

                            args.update(fixed)

                            ckpt = find_ckpt(name, args)
                            if ckpt is not None and ckpt.exists():
                                print(f"[SKIP] {name} task={task} ex={ex_name} split{split} {mode} "
                                      f"seed{seed} augment={augment} (ckpt exists: {ckpt.name})")
                                continue

                            jobs.append((name, script, args))

    print(f"Planned jobs: {len(jobs)}")

    if DRY_RUN:
        for name, script, args in jobs:
            print("[DRY]", script, args)
        return

    # --- Run sequentially ---
    for (name, script, args) in jobs:
        aug_tag = "aug" if args.get("augment") else "noaug"
        ex_tag = args.get("exercise-filter") or "all"
        tag = f"{name}_{args['task']}_{ex_tag}_{aug_tag}_split{args['split']}_{args['mode']}_seed{args['seed']}"
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
            print(f"[OK]  {tag} (log: {log_file})")

        time.sleep(1)

if __name__ == "__main__":
    main()
