#!/bin/bash
#SBATCH --job-name=SegDiffSample4       # Job name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks (one per GPU)
#SBATCH --cpus-per-task=3                 # CPU cores per task (GPU)
#SBATCH --gres=gpu:1                       # Request 2 GPUs
#SBATCH --mem=80GB                         # Total memory
#SBATCH --output=slurm/logs/segdiff_sample4.out    # Standard output log
#SBATCH --error=slurm/logs/segdiff_sample_4.err  # Standard error log
#SBATCH --partition=defq                   # Partition to submit to

CUDA_VISIBLE_DEVICES=0 mpiexec -n 1 python sample.py --model_path ./logs/2024-05-07-20-00-49-289417_astropath_12_0.0001_8_100_0.0_0/model60000.pt