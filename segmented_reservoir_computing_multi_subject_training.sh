#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --error=results/errors/segmented_reservoir_computing_multi_subject_training_%j.err
#SBATCH --output=results/outs/segmented_reservoir_computing_multi_subject_training_%j.out
srun python segmented_reservoir_computing_multi_subject_training.py --num_epochs 1000