This is the official repository of the paper [SegDiff: Image Segmentation with Diffusion Probabilistic Models](https://arxiv.org/abs/2112.00390)

The code is based on [Improved Denoising Diffusion Probabilistic Models.](https://github.com/openai/improved-diffusion)

## Installation
### Conda environment
To create the environment use the conda environment command
```
conda env create -f environment.yml
```

## Train and Evaluate
Execute the following commands (multi gpu is supported for training, set the gpus with CUDA_VISIBLE_DEVICES and -n for the actual number)

Training options:
```
# Training
--batch-size    Batch size
--lr            Learning rate

# Architecture
--rrdb_blocks       Number of rrdb blocks
--dropout           Dropout
--diffusion_steps   number of steps for the diffusion model

# Cityscapes
--class_name        name of class of cityscapes, options are ["bike", "bus", "person", "train", "motorcycle", "car", "rider"]
--expansion         boolean flag, for expansion setting or not

# Misc
--save_interval     interval for saving model weights
```

### Astropath

Training script example:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 mpiexec -n 4 python image_train_diff_medical.py --rrdb_blocks 12 --batch_size 2 --lr 0.0001 --diffusion_steps 100
```

Evaluation script example:
```
CUDA_VISIBLE_DEVICES=0 mpiexec -n 1 python image_sample_diff_medical.py --model_path path-for-model-weights
``` 

### Based on
```
@article{amit2021segdiff,
  title={Segdiff: Image segmentation with diffusion probabilistic models},
  author={Amit, Tomer and Nachmani, Eliya and Shaharbany, Tal and Wolf, Lior},
  journal={arXiv preprint arXiv:2112.00390},
  year={2021}
}
```
