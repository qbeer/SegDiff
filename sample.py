"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import datetime
import json
from pathlib import Path

import torch.distributed as dist

from improved_diffusion import dist_util, logger
from datasets.astropath import get_datasets
from improved_diffusion.sampling_util import sampling_major_vote_func
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.utils import set_random_seed
import warnings
warnings.filterwarnings('ignore')


def main():
    args = create_argparser().parse_args()

    original_logs_path = Path(args.model_path).parent
    logs_path = original_logs_path / f"{Path(args.model_path).stem}_major_vote"

    args.__dict__.update(json.loads((original_logs_path / 'args.json').read_text()))
    logger.info(args.__dict__)
    dist_util.setup_dist()

    logger.configure(dir=str(logs_path), log_suffix=f"val_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    if args.__dict__.get("seed") is None:
        seed = 1234
    else:
        seed = int(args.__dict__.get("seed"))
    set_random_seed(seed, deterministic=True)

    _, val_dataset = get_datasets(args)
    
    logger.log("sampling major vote val")
    (logs_path / "major_vote").mkdir(exist_ok=True)
    step = int(Path(args.model_path).stem.split("_")[-1].replace('model', ''))
    sampling_major_vote_func(diffusion, model, str(logs_path / "major_vote"), val_dataset, logger, args.clip_denoised,
                             step=step, n_rounds=len(val_dataset))

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
