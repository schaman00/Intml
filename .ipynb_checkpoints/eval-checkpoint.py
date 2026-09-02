import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Data.Dataset import MultiDSprites, DEFAULT_NPY_DIR, denormalize
from Model.Slot_attention_autoencoder import SlotAttentionAutoEncoder


def adjusted_rand_index(true_mask, pred_mask, eps=1e-8):
    n_points = true_mask.sum(dim=(1, 2))

    nij = torch.einsum('bnp,bnt->bpt', pred_mask, true_mask)
    a = nij.sum(dim=1)   # counts per true group   [B, G_true]
    b = nij.sum(dim=2)   # counts per pred group   [B, G_pred]

    rindex = (nij * (nij - 1)).sum(dim=(1, 2))
    aindex = (a * (a - 1)).sum(dim=1)
    bindex = (b * (b - 1)).sum(dim=1)

    expected = aindex * bindex / (n_points * (n_points - 1)).clamp(min=1.0)
    max_rindex = (aindex + bindex) / 2.0
    ari = (rindex - expected) / (max_rindex - expected + eps)

    single_group = (a > 0).sum(dim=1) <= 1
    return torch.where(single_group, torch.ones_like(ari), ari)


def ari_foreground(gt_masks, pred_masks):
    b = gt_masks.shape[0]
    true_fg = gt_masks[:, 1:].flatten(2).permute(0, 2, 1).float()   # [B, N, E-1]

    k = pred_masks.shape[1]
    winner = pred_masks.squeeze(-1).argmax(dim=1).flatten(1)        # [B, N]
    pred_onehot = F.one_hot(winner, k).float()                      # [B, N, K]

    return adjusted_rand_index(true_fg, pred_onehot)


def build_model(ckpt, device):
    resolution = (64, 64)
    num_slots = 6
    num_iterations=3
    hid_dim=64
    model = SlotAttentionAutoEncoder(
        resolution=resolution,
        num_slots=num_slots,
        num_iterations=num_iterations,
        hid_dim=hid_dim).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    mse_sum, ari_sum, n = 0.0, 0.0, 0
    for i, (image, gt_mask) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        image = image.to(device)
        gt_mask = gt_mask.to(device)
        recon, recons, masks, slots, attn = model(image)

        bs = image.shape[0]
        mse_sum += float(((recon - image) ** 2).mean(dim=(1, 2, 3)).sum())
        ari_sum += float(ari_foreground(gt_mask, masks).sum())
        n += bs
    return mse_sum / n, ari_sum / n, n


def colorize(label_map, n_labels):
    """[H, W] int -> [H, W, 3] float, using a fixed qualitative palette."""
    palette = plt.get_cmap('tab10')(np.arange(10))[:, :3]
    palette = np.concatenate([np.ones((1, 3)) * 0.92, palette])  # label 0 = grey
    return palette[np.clip(label_map, 0, n_labels) % len(palette)]


@torch.no_grad()
def qualitative_figure(model, dataset, device, out_path, n_examples=6, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(dataset), n_examples, replace=False)
    items = [dataset[int(i)] for i in idx]
    has_gt = isinstance(items[0], (tuple, list))
    images = torch.stack([it[0] if has_gt else it for it in items]).to(device)
    gt = torch.stack([it[1] for it in items]) if has_gt else None

    recon, recons, masks, slots, attn = model(images)
    k = masks.shape[1]
    pred_seg = masks.squeeze(-1).argmax(dim=1).cpu().numpy()      # [B, H, W]

    n_cols = k + (4 if has_gt else 2)
    fig, axes = plt.subplots(n_examples, n_cols,
                             figsize=(1.6 * n_cols, 1.6 * n_examples))
    if n_examples == 1:
        axes = axes[None, :]

    for r in range(n_examples):
        panels = [denormalize(images[r]).permute(1, 2, 0).cpu().numpy(),
                  denormalize(recon[r]).permute(1, 2, 0).cpu().numpy()]
        titles = ['input', 'recon']

        if has_gt:
            gt_labels = gt[r].float().argmax(dim=0).numpy() * gt[r].any(dim=0).numpy()
            panels.append(colorize(gt_labels, gt.shape[1]))
            titles.append('GT seg')
            panels.append(colorize(pred_seg[r] + 1, k + 1))
            titles.append('pred seg')

        for s in range(k):
            m = masks[r, s]                                  # [H, W, 1]
            slot_rgb = denormalize(recons[r, s].permute(2, 0, 1)).permute(1, 2, 0)
            panels.append((slot_rgb * m + (1 - m)).clamp(0, 1).cpu().numpy())
            titles.append('slot {}'.format(s + 1))

        for c, (panel, title) in enumerate(zip(panels, titles)):
            axes[r, c].imshow(panel)
            axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title(title, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote {}'.format(out_path))


def training_curves(run_dirs, out_path):
    import csv
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    plotted = False
    for label, run_dir in run_dirs:
        path = os.path.join(run_dir, 'log.csv')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        step = [int(r['step']) for r in rows]
        axes[0].plot(step, [float(r['loss']) for r in rows], label=label)
        axes[1].plot(step, [float(r['mask_entropy']) for r in rows], label=label)
        axes[2].plot(step, [int(r['slots_used']) for r in rows], label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    for ax, title in zip(axes, ['MSE loss', 'mask entropy', 'slots used']):
        ax.set_xlabel('step')
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_yscale('log')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote {}'.format(out_path))


def main():
    npy_dir = DEFAULT_NPY_DIR
    out_dir = ''
    model_path = ''
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    val_set = MultiDSprites(npy_dir, split='val', with_masks=True)
    loader = DataLoader(val_set, batch_size=32, shuffle=False,
                        num_workers=0)

    entries = [('slot_attention', model_path)]

    metrics, run_dirs = {}, []
    for name, path in entries:
        ckpt = torch.load(path, map_location=device)
        model = build_model(ckpt, device)
        mse, ari, n = evaluate(model, loader, device,)
        metrics[name] = {'checkpoint': path, 'step': ckpt['step'],'hid_dim': 64,
                          'num_slots': 6,
                         'val_images': n, 'mse': mse, 'ari_fg': ari}
        print('{:<18} MSE {:.5f}   ARI-FG {:.4f}   ({} val images)'.format(
            name, mse, ari, n))
        qualitative_figure(model, val_set, device,
                           os.path.join(out_dir, 'qualitative_{}.png'.format(name)),
                           n_examples=6)
        run_dirs.append((name, os.path.dirname(path)))

    training_curves(run_dirs, os.path.join(out_dir, 'curves.png'))

    out = os.path.join(out_dir, 'metrics.json')
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print('wrote {}'.format(out))


if __name__ == '__main__':
    main()
