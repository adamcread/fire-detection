#!/bin/bash

#SBATCH -N 1 # nodes
#SBATCH -c 1 # cores
#SBATCH --gres=gpu
#SBATCH -p "ug-gpu-small" # partition name
#SBATCH --qos="long-low-prio"
#SBATCH -t 12:00:00

source /etc/profile 
module load cuda/11.1
source env/bin/activate
python train_model.py
