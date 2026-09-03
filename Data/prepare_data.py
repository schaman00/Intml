import argparse
import os

import numpy as np

#Fom the terminal:

#python prepare_data.py

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    tfrecords = os.path.join(here, 'multi_dsprites_colored_on_colored.tfrecords')
    variant = 'colored_on_colored'
    out_dir = os.path.join(here, 'npy')
    n_train = 50000
    n_val = 2000
    
    if not os.path.exists(tfrecords):
        raise SystemExit(f'TFRecords not found: {tfrecords}')

    os.makedirs(out_dir, exist_ok=True)
    paths = {name: os.path.join(out_dir, name + '.npy') for name in
             ('train_images', 'val_images', 'val_masks', 'val_visibility')}

    if all(os.path.exists(p) for p in paths.values()):
        print('All outputs already exist, nothing to do.')
        for name, p in paths.items():
            print(f'  {name}: {np.load(p, mmap_mode="r").shape}')
        return

    from multi_object_datasets import multi_dsprites

    n_entities = multi_dsprites.MAX_NUM_ENTITIES[variant]
    h, w = multi_dsprites.IMAGE_SIZE
    n_total = n_train + n_val

    train_images = np.zeros((n_train, h, w, 3), dtype=np.uint8)
    val_images = np.zeros((n_val, h, w, 3), dtype=np.uint8)
    val_masks = np.zeros((n_val, n_entities, h, w), dtype=np.uint8)
    val_visibility = np.zeros((n_val, n_entities), dtype=np.float32)

    ds = multi_dsprites.dataset(tfrecords, variant).take(n_total)

    print(f'Reading {n_total} records from {tfrecords} ...')
    i = 0
    for rec in ds.as_numpy_iterator():
        img = rec['image']
        assert img.dtype == np.uint8, f'unexpected image dtype {img.dtype}'
        if i < n_train:
            train_images[i] = img
        else:
            j = i - n_train
            val_images[j] = img
            val_masks[j] = rec['mask'][..., 0]
            val_visibility[j] = rec['visibility']
        i += 1
        if i % 5000 == 0:
            print(f'  {i}/{n_total}')

    if i < n_total:
        raise SystemExit(f'dataset exhausted after {i} records, needed {n_total}')

    np.save(paths['train_images'], train_images)
    np.save(paths['val_images'], val_images)
    np.save(paths['val_masks'], val_masks)
    np.save(paths['val_visibility'], val_visibility)


    print(f'  train_images {train_images.shape}')
    print(f'  val_images   {val_images.shape}')
    print(f'  val_masks    {val_masks.shape}')



if __name__ == '__main__':
    main()
