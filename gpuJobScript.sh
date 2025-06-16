#! /bin/bash
#
#SBATCH -t 0-10:00:00
#SBATCH -N 1
#SBATCH --account=cavitation
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:2
#SBATCH --mail-user=naga@vt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=training

# Loading required modules for OpenFOAM and Postprocessing
module purge
module reset
module load GCC/13.3.0 Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
source "$HOME/workEnv/bin/activate"
python -u train.py --name raft-cloudcavitation  --stage cloudcavitation --validation cloudcavitation --restore_ckpt checkpoints/raft-sintel.pth --gpus 0 1 --num_steps 50000 --batch_size 6 --lr 0.0001 --image_size 192 640 --wdecay 0.00001 --gamma=0.85
#python evaluate.py --model=models/raft-sintel.pth --dataset=cloudcavitation
