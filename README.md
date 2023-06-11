# Basic Instructions on Taking the Challenge

1. Activate road++ conda virtual environment using
```bash
conda activate road++
```

2. There are some folders inside this folder, the dataset was chosen to be `Road Dataset` rather than `ROAD-Waymo dataset` only because it has affiliated papers.

3. Run the following command for training:
Official:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py /home/jianbingshen/compete/road++/ /home/jianbingshen/compete/road++/  /home/jianbingshen/compete/road++/RoadWaymoBaseline/kinetics-pt/ --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=RoadDataset --TEST_DATASET=RoadDataset --TRAIN_SUBSETS=train --VAL_SUBSETS=val --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041
```
Ours:
```
srun -p gpu_batch --gres=gpu:4 python main.py /home/jianbingshen/compete/road++/ /home/jianbingshen/compete/road++/  /home/jianbingshen/compete/road++/RoadWaymoBaseline/kinetics-pt/ --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=RoadDataset --TEST_DATASET=RoadDataset --TRAIN_SUBSETS=train --VAL_SUBSETS=val --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041
```
