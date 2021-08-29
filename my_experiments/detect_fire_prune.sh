#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5

cd ..
python detect.py --weights runs/train/exp30_0.5p_train/weights/pruning_best_pruning.pt --source data/images/fire \
--use-pruning
cd -