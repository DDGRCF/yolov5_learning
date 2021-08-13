#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5

cd ..
python detect.py --weights runs/train/exp31/weights/best.pt --source data/images/fire --use-pruning
cd -