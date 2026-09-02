import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.Encoder import Encoder
from Model.Decoder import Decoder
from Model.Slot_attention import SlotAttention

class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim, slot_mlp_dim=128):
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = tuple(resolution)
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = Encoder(self.resolution, hid_dim)
        self.decoder_cnn = Decoder(hid_dim, self.resolution)

        self.norm = nn.LayerNorm(hid_dim)
        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=hid_dim,
            iters=num_iterations,
            eps=1e-8,
            hidden_dim=slot_mlp_dim)

    def forward(self, image):
        x = self.encoder_cnn(image)          # [B, H*W, hid]
        x = self.norm(x)
        x = self.fc2(F.relu(self.fc1(x)))

        slots, attn = self.slot_attention(x)  # [B, K, hid], [B, K, H*W]

        x = self.decoder_cnn(slots)           # [B, K, H, W, 4]
        
        recons, masks = x.split([3, 1], dim=-1)
        masks = masks.softmax(dim=1)          # alpha compositing over slots

        recon_combined = (recons * masks).sum(dim=1).permute(0, 3, 1, 2)

        return recon_combined, recons, masks, slots, attn
