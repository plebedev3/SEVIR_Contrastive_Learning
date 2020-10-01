#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH --exclusive
#source /etc/profile

# Load the anaconda module
module load anaconda/2020a
#conda init bash
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2020a/etc/profile.d/conda.sh
conda activate rainymotion

python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --batch_size 32