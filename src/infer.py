import torch, numpy as np, cv2, sys
from ultralytics import YOLO
from train_seq import CNNBiLSTM, EXERCISES

def extract_kp_seq(video_path, model_name="yolov8n-pose.pt"):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)
    seq = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.predict(frame[:, :, ::-1], conf=0.25, verbose=False)
        if len(res)==0 or len(res[0].keypoints)==0:
            seq.append(np.zeros((17,2), dtype=np.float32))
            continue
        kp = res[0].keypoints.xy[0].cpu().numpy()  # (17,2)
        seq.append(kp)
    cap.release()
    if len(seq)==0: return None
    kp = np.stack(seq, axis=0) / 640.0
    feat = kp.reshape(kp.shape[0], -1)
    return torch.from_numpy(feat).unsqueeze(0).float()  # (1,T,34)

def predict(video_path, ckpt="outputs/models/seq_model.pt"):
    data = torch.load(ckpt, map_location="cpu")
    form_vocab = data["form_vocab"]
    model = CNNBiLSTM(in_dim=34, num_ex=len(EXERCISES), num_form=len(form_vocab))
    model.load_state_dict(data["model"]); model.eval()

    x = extract_kp_seq(video_path)
    with torch.no_grad():
        ex, fm = model(x)
    return EXERCISES[ex.argmax(1).item()], form_vocab[fm.argmax(1).item()]

if __name__ == "__main__":
    v = sys.argv[1]
    print(predict(v))
