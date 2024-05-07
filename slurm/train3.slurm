#!/bin/bash
#SBATCH --job-name=SegDiff3       # Job name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=4                         # Number of tasks (one per GPU)
#SBATCH --cpus-per-task=3                 # CPU cores per task (GPU)
#SBATCH --gres=gpu:4                       # Request 2 GPUs
#SBATCH --mem=120GB                         # Total memory
#SBATCH --output=slurm/logs/segdiff3_%j.out    # Standard output log
#SBATCH --error=slurm/logs/segdiff3_%j.err  # Standard error log
#SBATCH --partition=defq                   # Partition to submit to

CUDA_VISIBLE_DEVICES=0,1,2,3 mpiexec -n 4 python train.py --rrdb_blocks 12 --batch_size 8 --lr 0.0001 --diffusion_steps 100