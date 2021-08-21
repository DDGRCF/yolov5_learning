#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd ..
python train.py --epochs 80 --weights weights/yolov5l.pt --cfg models/yolov5l.yaml \
--data data/fire.yaml --batch-size 16 --workers 8 --device 0 --use-pruning --layer-rate 0.5  \
--layer-gap 0,321,3 --skip-downsample --pruning-method SFP
cd -
