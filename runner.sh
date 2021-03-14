# !/bin/bash

# SBATCH -N 1 # nodes
# SBATCH -c 4 # cores
# SBATCH -p "ug-gpu-small" # partition name
# SBATCH --qos="long-high-prio"
# SBATCH -t 02:00:00

source env/bin/activate

python package_tester.py
