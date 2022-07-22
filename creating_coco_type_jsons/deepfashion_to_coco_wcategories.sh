#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 96G
#SBATCH --time 5:00:00
#SBATCH --output=deepfashion_to_coco_wcategories_%j.log
#SBATCH --gres gpu:1

sleep 5
python3 deepfashionToCoco.py --dataset-root /scratch/izar/khayatan/deepfashion/ --root-save /scratch/izar/khayatan/deepfashion