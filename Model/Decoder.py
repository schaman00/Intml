import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Position_embed import PositionEmbed


class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution, slot_dim=None):
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = tuple(resolution)
        slot_dim = slot_dim if slot_dim is not None else hid_dim

        self.fc = nn.Linear(slot_dim, hid_dim)
        self.pos_embed = PositionEmbed(hid_dim, self.resolution)

        self.conv1 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.out = nn.Conv2d(hid_dim, 4, 3, padding=1)

    def forward(self, slots):
        b, k, _ = slots.shape
        h, w = self.resolution

        x = self.fc(slots)                                    
        x = x[:, :, None, None, :].expand(-1, -1, h, w, -1) 
        x = self.pos_embed(x)
        x = x.permute(0, 1, 4, 2, 3).reshape(b * k, self.hid_dim, h, w)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.out(x)

        return x.view(b, k, 4, h, w).permute(0, 1, 3, 4, 2)
