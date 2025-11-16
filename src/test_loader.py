# src/test_loader.py
from datasets.core import make_dataloaders

if __name__ == "__main__":
    # Run from the project root for simplest path handling:
    #   python src/test_loader.py
    train_dl, val_dl, test_dl = make_dataloaders(
        split_id=0,
        batch_size=2,
        num_workers=2,
        mode="rep",          # 'video' for full videos, 'rep' for 3-segment clips
        augment=True,
        allow_hflip=False,   # keep False if left/right matters
        num_frames=32,
        resize=224,
        include_pose=False
    )
    print("Batches in train:", len(train_dl))
    b = next(iter(train_dl))
    print("video:", b["video"].shape)         # [B, T, C, H, W]
    print("exercise_id:", b["exercise_id"].shape)
    print("form_id:", b["form_id"].shape)
    print("subtype_id:", b["subtype_id"].shape)
    print("file_stem:", b["file_stem"])
