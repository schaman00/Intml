import numpy as np
import torch
import torch.nn as nn


def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing='ij')
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))


class PositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.register_buffer('grid', build_grid(resolution), persistent=False)

    def forward(self, inputs):
        return inputs + self.embedding(self.grid.to(inputs.dtype))
