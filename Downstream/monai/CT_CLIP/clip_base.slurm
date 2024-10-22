#!/bin/bash

# NOTE: Lines starting with "#SBATCH" are valid SLURM commands or statements,
#       while those starting with "#" and "##SBATCH" are comments.

#SBATCH -J base

#SBATCH -t 72:00:00 #Maximum runtime of 48 hours

# Enable email notificaitons when job begins and ends
#SBATCH --mail-user=lwubf@connect.ust.hk #Update your email address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# Choose partition (queue) with "gpu"
#SBATCH -p large-project

# To use 24 cpu core and 1 gpu devices in a node
#SBATCH -N 1 -n 16 --gres=gpu:1

# Setup runtime environment if necessary
source ~/.bashrc
source activate nnunet

# Go to the job submission directory and run your application
cd /home/lwubf/SwinUNETR/CT_CLIP/
sh vocabfine_train_base.sh