#!/bin/bash
#SBATCH --partition deimos
#SBATCH -N 1
#SBATCH --job-name pendulum

source ~/miniconda3/etc/profile.d/conda.sh
source activate base

module load mpi/mpich/4.1.2-icc-oneapi2023.2-ch4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/APP/u22/x86/lib
export OMP_NUM_THREADS=32

python -u generate.py # visualize data