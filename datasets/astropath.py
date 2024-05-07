from __future__ import annotations

import multiprocessing
import os
from functools import partial
from pathlib import Path

from mpi4py import MPI

import numpy as np
import skimage as ski
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset

def get_datasets(args):
    dataset = AstropathDataset(
        args.data_root, database=args.database, level=args.level, layers=args.layers,
        augment=args.augment, test_run=args.test_run, mp_enabled=args.mp_enabled,
        progress=args.progress, image_size=256 if args.chunked else 1280,
        equalize=True
    )
    
    # NOTE: need to reinitialize the dataset for validation
    #       to be able to NOT augment the validation set
    test_dataset = AstropathDataset(
        args.data_root, database=args.database, level=args.level, layers=args.layers,
        augment=False, test_run=args.test_run, image_size=256 if args.chunked else 1280,
        # NOTE: add the same seed to make sure the same samples are in the validation set
        tile_paths=dataset.tile_paths, sample_ids=dataset.sample_ids,
        equalize=True
    )
    
    unique_sample_ids = np.unique(dataset.sample_ids, return_counts=False)

    train_unique_sample_ids, validation_unique_sample_ids = train_test_split(
        unique_sample_ids, test_size=0.15, random_state=args.seed,
    )

    train_idx = np.where(np.isin(dataset.sample_ids, train_unique_sample_ids))[0]
    validation_idx = np.where(np.isin(dataset.sample_ids, validation_unique_sample_ids))[0]

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(test_dataset, validation_idx)

    train_ids = {dataset.sample_ids[idx] for idx in train_idx}
    valid_ids = {test_dataset.sample_ids[idx] for idx in validation_idx}
    assert train_ids.isdisjoint(valid_ids), 'Train and validation sets are not disjoint'

    return train_dataset, valid_dataset

def load_data(args, deterministic=False):
    dataset, _ = get_datasets(args)
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader

def filter_tile(tile_path, level, minimum_overlap=0.01):
    roi_image_path = tile_path.name.replace(f'L{level}.npy', f'roi_tissue_anno_L{level}.png')
    roi_image_path = os.path.join(tile_path.parent, roi_image_path)
    roi_image = ski.io.imread(roi_image_path)

    roi_mask = np.zeros_like(roi_image)
    roi_mask[roi_image > 0] += 1

    tumor_image_path = tile_path.name.replace(f'L{level}.npy', f'tumor_anno_L{level}.png')
    tumor_image_path = os.path.join(tile_path.parent, tumor_image_path)
    tumor_image = ski.io.imread(tumor_image_path)

    tumor_mask = np.zeros_like(tumor_image)
    tumor_mask[tumor_image > 0] += 1

    # NOTE: if both the roi mask and the tumor mask
    #       is less then 1% of the image
    #       discard the tile
    if (roi_mask.sum() / np.prod(roi_mask.shape)) < minimum_overlap and\
            (tumor_mask.sum() / np.prod(tumor_mask.shape)) < minimum_overlap:
        return None
    else:
        return tile_path



class AstropathDataset(Dataset):
    def __init__(
        self, data_base_path, database, level, image_size=1280,
        layers=[0], augment=False, pool_size=8, test_run=False,
        tile_paths=None, sample_ids=None, mp_enabled=False,
        progress=False, equalize=False
    ):
        self.database = database
        self.level = level
        self.layers = layers
        self.image_size = image_size
        self.augment = augment
        self.equalize = equalize
        
        self.mean = torch.from_numpy(np.array([127.5 for _ in range(len(layers))]))
        self.std = torch.from_numpy(np.array([127.5 for _ in range(len(layers))]))

        if tile_paths is not None and sample_ids is not None:
            self.tile_paths = tile_paths
            self.sample_ids = sample_ids
            
            self.tile_paths = self.tile_paths
            self.sample_ids = self.sample_ids

            return

        if self.image_size != 1280:
            path = Path(data_base_path) / f'{database}_chunks' / str(level)
        else:
            path = Path(data_base_path) / database / str(level)

        tile_paths = list(path.glob('*.npy'))

        if test_run:
            tile_paths = tile_paths[:256]

        filter_func = partial(filter_tile, level=level)

        if mp_enabled:
            with multiprocessing.Pool(pool_size) as pool:
                filtered_tile_paths = []
                jobs = pool.imap(filter_func, tile_paths)
                pbar = tqdm(jobs, total=len(tile_paths), desc='Filtering empty ROI masks') if progress else jobs
                for path in pbar:
                    if path is not None:
                        filtered_tile_paths.append(path)
        else:
            filtered_tile_paths = []
            pbar = tqdm(tile_paths, desc='Filtering empty ROI masks') if progress else tile_paths
            for path in pbar:
                path = filter_func(path)
                if path is not None:
                    filtered_tile_paths.append(path)

        filtered_tile_paths = sorted(filtered_tile_paths, key=lambda x: x.stem.split('_')[0])

        self.tile_paths = filtered_tile_paths

        # NOTE: define sample_ids for stratified sampling
        self.sample_ids = list(map(lambda x: x.stem.split('_')[0], self.tile_paths))
        
        shard = MPI.COMM_WORLD.Get_rank()
        num_shards = MPI.COMM_WORLD.Get_size()
        
        self.tile_paths = self.tile_paths[shard::num_shards]
        self.sample_ids = self.sample_ids[shard::num_shards]

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        tile = np.load(tile_path)

        tile_name = tile_path.stem

        input_tile = tile[:, :, self.layers] / 255.0

        if self.equalize:
            for layer_ind in range(len(self.layers)):
                input_tile[:, :, layer_ind] = ski.exposure.equalize_adapthist(
                    input_tile[:, :, layer_ind], clip_limit=0.03, kernel_size=8, nbins=256,
                )
                
        roi_image_path = tile_path.name.replace(f'L{self.level}.npy', f'roi_tissue_anno_L{self.level}.png')
        roi_image_path = os.path.join(tile_path.parent, roi_image_path)
        roi_image = ski.io.imread(roi_image_path)

        roi_mask = np.zeros_like(roi_image)
        roi_mask[roi_image > 0] += 255

        tumor_image_path = tile_path.name.replace(f'L{self.level}.npy', f'tumor_anno_L{self.level}.png')
        tumor_image_path = os.path.join(tile_path.parent, tumor_image_path)
        tumor_image = ski.io.imread(tumor_image_path)

        tumor_mask = np.zeros_like(tumor_image)
        tumor_mask[tumor_image > 0] += 255
        
        valid_mask = (tumor_mask | roi_mask) > 0
        invalid_mask = ~valid_mask

        input_tile[invalid_mask] = 0
        
        if self.image_size != 256:
            input_tile = ski.transform.resize(input_tile, (self.image_size, self.image_size), order=1)
            tumor_mask = ski.transform.resize(tumor_mask, (self.image_size, self.image_size), order=0)
            roi_mask = ski.transform.resize(roi_mask, (self.image_size, self.image_size), order=0)

        # NOTE: pad the image if it's smaller than the desired size
        #       this happens with patches on the 'edge' of the WSIs
        if tumor_mask.shape[0] != self.image_size or tumor_mask.shape[1] != self.image_size:
            pad_x = self.image_size - tumor_mask.shape[0]
            pad_y = self.image_size - tumor_mask.shape[1]

            input_tile = np.pad(input_tile, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')
            tumor_mask = np.pad(tumor_mask, ((0, pad_x), (0, pad_y)), mode='constant')
            roi_mask = np.pad(roi_mask, ((0, pad_x), (0, pad_y)), mode='constant')

        if self.augment:
            h_flip = np.random.rand() > 0.5
            v_flip = np.random.rand() > 0.5
            if h_flip:
                input_tile = np.flip(input_tile, axis=1)
                tumor_mask = np.flip(tumor_mask, axis=1)
                roi_mask = np.flip(roi_mask, axis=1)
            if v_flip:
                input_tile = np.flip(input_tile, axis=0)
                tumor_mask = np.flip(tumor_mask, axis=0)
                roi_mask = np.flip(roi_mask, axis=0)

            do_crop = np.random.rand() > 0.5
            if do_crop:
                from_x = np.random.randint(0, 3 * self.image_size // 10)
                from_y = np.random.randint(0, 3 * self.image_size // 10)

                to_x = np.random.randint(7 * self.image_size // 10, self.image_size)
                to_y = from_y + (to_x - from_x)

                _tile = input_tile[from_x:to_x, from_y:to_y, :]
                _tumor_mask = tumor_mask[from_x:to_x, from_y:to_y]
                _roi_mask = roi_mask[from_x:to_x, from_y:to_y]

                # NOTE: check if there is any information in the cropped image
                if np.all(np.unique(_roi_mask) == 0):
                    pass  # NOTE: skip the cropping
                else:
                    input_tile = ski.transform.resize(_tile, (self.image_size, self.image_size), order=1)
                    tumor_mask = ski.transform.resize(_tumor_mask, (self.image_size, self.image_size), order=0)
                    roi_mask = ski.transform.resize(_roi_mask, (self.image_size, self.image_size), order=0)

        min_x, max_x = valid_mask.nonzero()[0].min(), valid_mask.nonzero()[0].max()
        min_y, max_y = valid_mask.nonzero()[1].min(), valid_mask.nonzero()[1].max()

        bbox = [min_x, min_y, max_x, max_y]
        bbox = torch.from_numpy(np.array(bbox)).float()

        tumor_mask = tumor_mask / 255.0
        roi_mask = roi_mask / 255.0

        # NOTE: normalize the input and output tiles to [-1, 1]
        input_tile = 2. * ( input_tile.astype(np.float32) - 0.5 )
        tumor_mask = tumor_mask.astype(np.float32)
        roi_mask = roi_mask.astype(np.float32)
        
        tumor_mask = 2. * ( tumor_mask - 0.5 )
        tumor_mask = torch.from_numpy(tumor_mask[:, :, None]).permute(2, 0, 1).float()

        input_tile = torch.from_numpy(input_tile).permute(2, 0, 1).float()
        output_dict = { "conditioned_image": input_tile }

        return tumor_mask, output_dict, f"{tile_name}_{idx}"
