import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Data.Dataset import MultiDSprites, DEFAULT_NPY_DIR
from Model.Slot_attention_autoencoder import SlotAttentionAutoEncoder


def lr_at(step, base_lr, warmup_steps, decay_rate, decay_steps):
    warmup = min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
    return base_lr * warmup * (decay_rate ** (step / decay_steps))


def mask_diagnostics(masks):
    p = masks.detach().squeeze(-1).float()                       # [B, K, H, W]
    entropy = -(p * (p + 1e-8).log()).sum(dim=1).mean()
    winner = p.argmax(dim=1)                            # [B, H, W]
    k = p.shape[1]
    frac = torch.stack([(winner == i).float().mean() for i in range(k)])
    slots_used = int((frac >= 0.01).sum())
    return float(entropy), slots_used


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scaler(enabled):
    try: 
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def main():
    seed=0
    set_seed(seed)

    out = 'Out10'
    npy_dir = DEFAULT_NPY_DIR
    os.makedirs(out, exist_ok=True)


    batch_size=16
    steps=17000
    log_every=200
    ckpt_every=2000
    num_workers=2
    resolution = (64, 64)
    num_slots = 6
    num_iterations=1
    hid_dim=64
    
    warmup_steps=2500
    decay_steps=8000
    decay_rate=0.5
    clip_grad = 1
    base_lr = 4e-4
    
    resume = None
    no_amp = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = (not no_amp) and device.type == 'cuda'

    torch.backends.cudnn.benchmark = True


    train_set = MultiDSprites(npy_dir, split='train')
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers,
                        pin_memory=(device.type == 'cuda'), drop_last=True,
                        persistent_workers=num_workers > 0)

    model = SlotAttentionAutoEncoder(
        resolution=tuple(resolution),
        num_slots=num_slots,
        num_iterations=num_iterations,
        hid_dim=hid_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scaler = make_scaler(use_amp)

    start_step, best_loss = 0, float('inf')
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])
        start_step = ckpt['step'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print('resumed from {} at step {}'.format(resume, start_step))

    log_path = os.path.join(out, 'log.csv')
    log_file = open(log_path, 'a', newline='')
    writer = csv.writer(log_file)
    if os.path.getsize(log_path) == 0:
        writer.writerow(['step', 'loss', 'lr', 'mask_entropy', 'slots_used',
                         'recon_min', 'recon_max', 'sec_per_step'])

    def save(step, loss, name):
        torch.save({'step': step, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if use_amp else None,
                    'best_loss': best_loss, 'loss': loss,
                    },
                   os.path.join(out, name))

    model.train()
    stream = infinite(loader)
    running, t0 = 0.0, time.time()
    max_entropy = math.log(num_slots)
    loss_value = float('nan')

    for step in range(start_step, steps):
        image = next(stream).to(device, non_blocking=True)

        current_lr = lr_at( step=step, base_lr=base_lr, warmup_steps=warmup_steps, decay_rate=decay_rate,decay_steps=decay_steps,)

        for group in optimizer.param_groups:
            group['lr'] = current_lr

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            recon_combined, recons, masks, slots, attn = model(image)
            loss = criterion(recon_combined, image)


        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if clip_grad > 0:                     
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise SystemExit('loss became {} at step {} -- aborting'.format(
                loss_value, step))
        running += loss_value

        if (step + 1) % log_every == 0:
            with torch.no_grad():
                entropy, slots_used = mask_diagnostics(masks)
            mean_loss = running / log_every
            running = 0.0
            sec = (time.time() - t0) / log_every
            t0 = time.time()
            rc = recon_combined.detach()
            r_min, r_max = float(rc.min()), float(rc.max())
            print('step {:>7}/{}  loss {:.5f}  lr {:.2e}  H(mask) {:.3f}/{:.3f}  '
                  'slots_used {}/{}  recon [{:+.2f},{:+.2f}]  {:.0f} ms/step'.format(
                      step + 1, steps, mean_loss, current_lr, entropy, max_entropy,
                      slots_used, num_slots, r_min, r_max, sec * 1000))
            writer.writerow([step + 1, mean_loss, current_lr, entropy, slots_used,
                             r_min, r_max, sec])
            log_file.flush()

            if mean_loss < best_loss:
                best_loss = mean_loss
                save(step, mean_loss, 'best.pt')

        if (step + 1) % ckpt_every == 0:
            save(step, loss_value, '{}_step_model.pt'.format(step + 1))

    save(steps - 1, loss_value, 'last.pt')
    log_file.close()
    print('\nDone. best loss {:.5f}. Checkpoints in {}'.format(best_loss, out))


if __name__ == '__main__':
    main()
