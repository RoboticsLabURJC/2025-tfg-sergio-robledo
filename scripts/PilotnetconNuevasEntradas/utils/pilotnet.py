import torch
import torch.nn as nn
import torch.nn.functional as F

class PilotNet(nn.Module):
    def __init__(self, image_shape=(66, 200, 3), num_labels=2):
        super().__init__()
        # Convs como ya las tienes
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # --- calcula flatten_dim con un dummy ---
        c, h, w = 3, image_shape[0], image_shape[1]
        with torch.no_grad():
            x = torch.zeros(1, c, h, w)
            x = self._forward_features(x)
            flatten_dim = x.view(1, -1).shape[1] 

        # === CAMBIO CLAVE: +1 por la velocidad ===
        self.fc1 = nn.Linear(flatten_dim + 1, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc_out = nn.Linear(10, num_labels)

    def _forward_features(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        return x

    # ===forward SIEMPRE recibe (img, speed) ===
    def forward(self, img, speed):
        # img: (B,3,66,200)
        # speed: (B,1) escalar normalizado por muestra
        x = self._forward_features(img)
        x = x.view(x.size(0), -1)          # (B, flatten_dim)
        x = torch.cat([x, speed], dim=1)   # (B, flatten_dim+1)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        out = self.fc_out(x)               # (B, 2) -> steer, throttle
        return out
