import numpy as np
import os
from torchmetrics.classification import BinaryAUROC
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import Dice

from improved_diffusion.metrics import FBound_metric
from improved_diffusion.metrics import WCov_metric

import matplotlib.pyplot as plt
import torch
from PIL import Image

def diff_image(pred_mask, true_mask):
    cmap = plt.get_cmap('seismic')

    pred_mask = np.round(pred_mask)
    error_mask = np.log2((pred_mask + 1) / (true_mask + 1))
    error_mask = np.clip(error_mask, -1, 1)
    # NOTE: this way, 0 becomes 0.5, 1 becomes 1, -1 becomes 0
    #       cmap will map 0.5 to white, 1 to red, 0 to blue
    #       red will indicite false positive, blue will
    #       indicate false negative
    error_mask = (error_mask + 1) / 2

    error_mask_is_not_matching = (error_mask != 0.5)

    pred_mask_matching_within_true_mask = ((pred_mask.astype(int) == 1) & (true_mask.astype(int) == 1))

    diff = cmap(error_mask[..., 0])[..., :3] * error_mask_is_not_matching

    # NOTE: make white where matching completely
    diff[pred_mask_matching_within_true_mask[..., 0], :] = [1, 1, 1]

    return diff

def reconstruct(reconstructions, output_dir, unique_sample_ids):
    whole_slide_level_metrics = {
        'WCov': {'mean': [], 'std': None},
        'FBound': {'mean': [], 'std': None},
        'F1': {'mean': [], 'std': None},
        'AUROC': {'mean': [], 'std': None},
        'Precision': {'mean': [], 'std': None},
        'Recall': {'mean': [], 'std': None},
        'Dice': {'mean': [], 'std': None},
        'Jaccard': {'mean': [], 'std': None},
    }

    for sample_id in unique_sample_ids:
        pred_masks = [
            reconstructions['pred_mask'][idx]
            for idx, _sample_id in enumerate(reconstructions['sample_id']) if _sample_id == sample_id
        ]
        true_masks = [
            reconstructions['true_mask'][idx]
            for idx, _sample_id in enumerate(reconstructions['sample_id']) if _sample_id == sample_id
        ]
        images = [
            reconstructions['images'][idx]
            for idx, _sample_id in enumerate(reconstructions['sample_id']) if _sample_id == sample_id
        ]

        x = [
            reconstructions['x'][idx] for idx, _sample_id in enumerate(
                reconstructions['sample_id'],
            ) if _sample_id == sample_id
        ]
        y = [
            reconstructions['y'][idx] for idx, _sample_id in enumerate(
                reconstructions['sample_id'],
            ) if _sample_id == sample_id
        ]
        x, y = np.array(x), np.array(y)

        width = [
            reconstructions['width'][idx]
            for idx, _sample_id in enumerate(reconstructions['sample_id']) if _sample_id == sample_id
        ]
        height = [
            reconstructions['height'][idx]
            for idx, _sample_id in enumerate(reconstructions['sample_id']) if _sample_id == sample_id
        ]
        width, height = np.array(width), np.array(height)

        min_x, max_x = min(x), max(x)
        min_y, max_y = min(y), max(y)

        x_range = max_x - min_x + max(width)
        y_range = max_y - min_y + max(height)

        reconstruction_mask = np.zeros((
            np.round(y_range * 256).astype(int),
            np.round(4 * x_range * 256).astype(int), 3,
        ))  # NOTE: 3 -> mask, pred, diff
        x, y = x - min_x, y - min_y
        x, y = x * 256, y * 256
        x, y = np.round(x).astype(int), np.round(y).astype(int)

        for (PR, GT, IM, _x, _y, _w, _h) in zip(pred_masks, true_masks, images, x, y, width, height):
            true_mask = GT
            pred_mask = PR
            image = IM

            current_width = _w * 256
            current_height = _h * 256

            reconstruction_mask[
                _y:(_y + current_height), _x:(_x + current_width),
            ] += np.repeat(true_mask[:current_height, :current_width], 3, axis=-1)

            pred_mask = np.round(pred_mask)

            reconstruction_mask[
                _y:(_y + current_height), np.round(x_range * 256).astype(int)
                + _x:np.round(x_range * 256).astype(int) + _x + current_width,
            ] += np.repeat(pred_mask[:current_height, :current_width], 3, axis=-1)

            diff = diff_image(pred_mask, true_mask)

            reconstruction_mask[
                _y:(_y + current_height), np.round(2 * x_range * 256).astype(int)
                + _x:np.round(2 * x_range * 256).astype(int) + _x + current_width,
            ] += diff[:current_height, :current_width]

            reconstruction_mask[
                _y:(_y + current_height), np.round(3 * x_range * 256).astype(int) + _x:np.round(
                    3 * x_range * 256,
                ).astype(int) + _x + current_width,
            ] += (0.7 * image[..., -1:] + 0.3 * diff)[:current_height, :current_width]

        # NOTE: split the image into 2 parts and stack them
        #       horizontally, this serves for visual inspection
        first, second = np.array_split(reconstruction_mask, 2, axis=1)
        reconstruction_mask = np.concatenate((first, second), axis=0)

        # NOTE: evaluate the full reconstructed masks
        gt_mask, pred_mask = np.array_split(first, 2, axis=1)
        gt_mask = gt_mask[..., 0]
        pred_mask = pred_mask[..., 0]

        wcov = WCov_metric(gt_mask, pred_mask)
        fbound = FBound_metric(gt_mask, pred_mask)

        f1 = BinaryF1Score()(torch.tensor(pred_mask).unsqueeze(0), torch.tensor(gt_mask).unsqueeze(0).long())
        jaccard = BinaryJaccardIndex()(
            torch.tensor(pred_mask).unsqueeze(0),
            torch.tensor(gt_mask).unsqueeze(0).long(),
        )
        auroc = BinaryAUROC()(
            torch.tensor(pred_mask).unsqueeze(0),
            torch.tensor(gt_mask).unsqueeze(0).long(),
        )
        precision = BinaryPrecision()(
            torch.tensor(pred_mask).unsqueeze(0),
            torch.tensor(gt_mask).unsqueeze(0).long(),
        )
        recall = BinaryRecall()(
            torch.tensor(pred_mask).unsqueeze(0),
            torch.tensor(gt_mask).unsqueeze(0).long(),
        )
        dice = Dice()(
            torch.tensor(pred_mask).unsqueeze(0),
            torch.tensor(gt_mask).unsqueeze(0).long(),
        )

        with open(os.path.join(output_dir, f'reconstruction_{sample_id}.yaml'), 'w') as f:
            f.write(f'WCov: {wcov}\n')
            f.write(f'FBound: {fbound}\n')
            f.write(f'F1: {f1}\n')
            f.write(f'Jaccard: {jaccard}\n')
            f.write(f'AUROC: {auroc}\n')
            f.write(f'Precision: {precision}\n')
            f.write(f'Recall: {recall}\n')
            f.write(f'Dice: {dice}\n')

        reconstruction_mask = (reconstruction_mask * 255).astype(np.uint8)
        img = Image.fromarray(reconstruction_mask, 'RGB')
        img.save(os.path.join(output_dir, f'reconstruction_{sample_id}.png'))

        whole_slide_level_metrics['WCov']['mean'].append(wcov)
        whole_slide_level_metrics['FBound']['mean'].append(fbound)
        whole_slide_level_metrics['F1']['mean'].append(f1)
        whole_slide_level_metrics['AUROC']['mean'].append(auroc)
        whole_slide_level_metrics['Precision']['mean'].append(precision)
        whole_slide_level_metrics['Recall']['mean'].append(recall)
        whole_slide_level_metrics['Dice']['mean'].append(dice)
        whole_slide_level_metrics['Jaccard']['mean'].append(jaccard)

    for _, metric in whole_slide_level_metrics.items():
        metric['std'] = np.std(metric['mean']).tolist()
        metric['mean'] = np.mean(metric['mean']).tolist()

    return whole_slide_level_metrics

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--level', type=int, default=3)
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    reconstructions = {'pred_mask': [], 'true_mask': [], 'images': [], 'x': [], 'y': [], 'width': [], 'height': [], 'sample_id': []}
    unique_sample_ids = []
    
    for filename in os.listdir(args.data_dir):
        if filename.endswith('_model_output.png'):
            tile_loc = filename.replace('_gt.png', '').replace('_model_output.png', '')
            tile_loc = tile_loc.replace('tile_', '').replace(f'_L{args.level}', '')
            tile_loc = tile_loc.replace('chunk_', '')

            sample_id, *tile_loc, _ = tile_loc.split('_')

            if len(tile_loc) == 4:
                # NOTE: format 0_0_(1-4)_(0-2) -> ['0', '0', '(1-4)', '(0-2)']
                x, y, x_loc, y_loc = tile_loc
                x, y = int(x), int(y)

                x_loc = [int(val) for val in x_loc.replace('(', '').replace(')', '').split('-')]
                y_loc = [int(val) for val in y_loc.replace('(', '').replace(')', '').split('-')]

                tile_loc = {
                    'x': x_loc[0] + x,
                    'y': y_loc[0] + y,
                    'width': 1,
                    'height': 1,
                }
                
            elif len(tile_loc) == 2:
                # NOTE: format (1-4)_(0-2) -> ['(1-4)', '(0-2)']
                x_loc, y_loc = tile_loc

                x_loc = [int(val) for val in x_loc.replace('(', '').replace(')', '').split('-')]
                y_loc = [int(val) for val in y_loc.replace('(', '').replace(')', '').split('-')]

                tile_loc = {
                    # NOTE: indicate with -1 that the tile is not divided
                    'x': x_loc[0],
                    'y': y_loc[0],
                    'width': x_loc[1] - x_loc[0],
                    'height': y_loc[1] - y_loc[0],
                }
                
            reconstructions['sample_id'].append(sample_id)
            unique_sample_ids.append(sample_id)
            reconstructions['width'].append(tile_loc['width'])
            reconstructions['height'].append(tile_loc['height'])
            reconstructions['x'].append(tile_loc['x'])
            reconstructions['y'].append(tile_loc['y'])
            
            pred_mask = Image.open(os.path.join(args.data_dir, filename)).convert('L')
            true_mask = Image.open(os.path.join(args.data_dir, filename.replace('_model_output.png', '_gt.png'))).convert('L')
            image = Image.open(os.path.join(args.data_dir, filename.replace('_model_output.png', '_condition_on.png')))
            
            pred_mask = np.array(pred_mask)[:, :, None]
            true_mask = np.array(true_mask)[:, :, None]
            image = np.array(image)[:,:,0][:,:,None]
            
            pred_mask = np.round(pred_mask / 255) - 1
            true_mask = np.round(true_mask / 255) - 1
            image = image / 255
            
            print(np.unique(pred_mask))
            print(np.unique(true_mask))
            
            reconstructions['pred_mask'].append(-pred_mask.astype(float))
            reconstructions['true_mask'].append(-true_mask.astype(float))
            reconstructions['images'].append(image)
    
    unique_sample_ids = list(set(unique_sample_ids))
    
    reconstruction_metrics = reconstruct(reconstructions, args.output_dir, unique_sample_ids)