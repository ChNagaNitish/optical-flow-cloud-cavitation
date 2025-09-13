#! /bin/bash
#
#SBATCH -t 0-01:00:00
#SBATCH -N 1
#SBATCH --ntasks 1
#SBATCH --account=cavitation
#SBATCH -p normal_q
#SBATCH --mail-user=naga@vt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=postProcess

# Loading required modules for OpenFOAM and Postprocessing
module purge
module reset
module load Python/3.12.3-GCCcore-13.3.0 FFmpeg/7.0.2-GCCcore-13.3.0
source ~/workEnvCPU/bin/activate

#python3 quiverVideo.py --path 32_50f.avi --velocity 32_50f_raft-cloudcav.h5 --fps 10
#python3 postProcessVelData.py --method vLinesAvg --path 32_50f_raft-cloudcav.h5
#python3 postProcessVelData.py --method hline --path 32_50f_raft-cloudcav.h5
python3 postProcessVelData.py --method points --path 32_50f_raft-cloudcav.h5