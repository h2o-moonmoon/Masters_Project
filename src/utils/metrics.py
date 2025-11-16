# src/utils/metrics.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def save_confusion(y_true, y_pred, class_names, out_path, normalize=True, title="Confusion"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_plot = cm.astype(float)
    if normalize and cm.sum(axis=1, keepdims=True).sum() > 0:
        with np.errstate(all='ignore'):
            cm_plot = cm_plot / np.maximum(cm_plot.sum(axis=1, keepdims=True), 1e-9)
            cm_plot = np.nan_to_num(cm_plot)

    # --- plot ---
    plt.figure(figsize=(5,4))
    plt.imshow(cm_plot, interpolation='nearest')
    plt.title(title)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=160)
    plt.close()

    # --- raw saves ---
    np.save(out_path.with_suffix(".npy"), cm)                       # raw counts
    np.savetxt(out_path.with_suffix(".csv"), cm, fmt="%d", delimiter=",")  # counts CSV
