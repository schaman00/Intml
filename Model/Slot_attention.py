import torch
import torch.nn as nn


class SlotAttention(nn.Module):
    def __init__(self, num_slots, hidden_dim, dim, iters=3, eps=1e-8):
        super().__init__()
        self.iters = iters
        self.num_slots = num_slots
        self.eps = eps
        self.scale = dim ** -0.5
        self.hidden_dim = hidden_dim

        self.slots_mu = nn.Parameter(torch.empty(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, dim))

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):

        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        with torch.autocast(device_type=inputs.device.type, enabled=False):
            inputs = inputs.float()

            mu = self.slots_mu.expand(b, n_s, -1)
            sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
            slots = mu + sigma * torch.randn(mu.shape, device=inputs.device,
                                             dtype=inputs.dtype)

            inputs = self.norm_input(inputs)
            k, v = self.to_k(inputs), self.to_v(inputs)

            attn = None
            for _ in range(self.iters):
                slots_prev = slots

                slots = self.norm_slots(slots)
                q = self.to_q(slots)

                dots = torch.einsum('bid,bjd->bij', q, k) * self.scale

                softmax_dim = 1
                attn = dots.softmax(dim=softmax_dim)

                weights = attn + self.eps
                weights = weights / weights.sum(dim=-1, keepdim=True)
                updates = torch.einsum('bjd,bij->bid', v, weights)

                slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))
                slots = slots.reshape(b, -1, d)
                slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn