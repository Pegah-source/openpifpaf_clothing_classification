#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 96G
#SBATCH --time 72:00:00
#SBATCH --output=classification_clothing_0001_%j.log
#SBATCH --gres gpu:2

sleep 5
python3 -m openpifpaf.train --dataset deepfashion --basenet=shufflenetv2k30 \
  --lr=0.0001 --momentum=0.95  --b-scale=5.0 --epochs=300 --lr-decay 160 260 --lr-decay-epochs=10 \
  --weight-decay=1e-5 --weight-decay=1e-5  --val-interval 10 --loader-workers 16 --batch-size 8