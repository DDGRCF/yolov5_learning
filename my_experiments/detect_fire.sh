#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd .. 
python detect.py --weights runs/train/exp18/weights/best.pt \
--source data/images/fire --device 1