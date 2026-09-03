import os

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_NPY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'npy')


class MultiDSprites(Dataset):
    def __init__(self, npy_dir=DEFAULT_NPY_DIR, split='train', with_masks=False):
        if split not in ('train', 'val'):
            raise ValueError(f'split must be "train" or "val", got {split!r}')
        if with_masks and split != 'val':
            raise ValueError('masks were only exported for the val split')

        self.images_path = os.path.join(npy_dir, f'{split}_images.npy')
        if not os.path.exists(self.images_path):
            raise FileNotFoundError(
                f'{self.images_path} not found. Run:\n'
                '    python prepare_data.py '
                )

        self.images = np.load(self.images_path, mmap_mode='r')
        self.masks = None
        if with_masks:
            self.masks = np.load(os.path.join(npy_dir, 'val_masks.npy'), mmap_mode='r')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.asarray(self.images[idx], dtype=np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1) 
        if self.masks is None:
            return img
        mask = torch.from_numpy(np.asarray(self.masks[idx]) > 127)
        return img, mask


def denormalize(x):
    return (x.clamp(-1, 1) + 1.0) / 2.0
