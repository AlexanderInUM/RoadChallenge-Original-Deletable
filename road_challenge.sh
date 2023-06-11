#!/usr/bin/env bash

#SBATCH --partition=gpu_batch
#SBATCH --job-name=road_challenge
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gpu09
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00

set -x

srun -p gpu_batch --gres=gpu:4 python main.py /home/jianbingshen/compete/road++/ /home/jianbingshen/compete/road++/ /home/jianbingshen/compete/road++/RoadWaymoBaseline/kinetics-pt/ --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=RoadDataset --TEST_DATASET=RoadDataset --TRAIN_SUBSETS=train --VAL_SUBSETS=val --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041 
