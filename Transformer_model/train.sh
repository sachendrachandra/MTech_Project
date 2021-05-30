#!/bin/sh
#SBATCH --job-name=test_job
#SBATCH --ntasks=1 
#SBATCH --output=test_job%j.out
#SBATCH --gres=gpu:2
#SBATCH --partition=q2h_48h-2G
python3 ../run.py
pwd; hostname; date | tee results

