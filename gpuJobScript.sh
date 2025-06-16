#! /bin/bash
#
#SBATCH -t 0-3:00:00
#SBATCH -N 1
#SBATCH --account=cavitation
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:1
#SBATCH --mail-user=naga@vt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=training

# Loading required modules
module purge
module reset
module load GCC/13.3.0 Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
# Activating the Python Environment
source "$HOME/workEnv/bin/activate"
# Running the Optical Flow code
python tracking.py --method=raft --model=models/raft-sintel.pth --path=/home/naga/experiment/48.avi
