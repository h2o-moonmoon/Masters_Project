import torch
import torch.nn as nn

class PoseGRU_MTL(nn.Module):
    def __init__(self, J=17, D=2, H=128, exercise_classes=5, form_classes=2, type_classes=3, layers=1, bidir=False):
        super().__init__()
        self.gru = nn.GRU(input_size=J*D, hidden_size=H, num_layers=layers,
                          batch_first=True, bidirectional=bidir)
        out_dim = H * (2 if bidir else 1)
        self.head_ex = nn.Linear(out_dim, exercise_classes)
        self.head_form = nn.Linear(out_dim, form_classes)
        self.head_type = nn.Linear(out_dim, type_classes)

    def forward(self, pose):  # pose: [B,T,J,2]
        B,T,J,D = pose.shape
        x = pose.view(B,T,J*D)
        _, h = self.gru(x)          # h: [layers*(2?), B, H]
        z = h[-1]                    # [B,H] (use last layer)
        return self.head_ex(z), self.head_form(z), self.head_type(z)
