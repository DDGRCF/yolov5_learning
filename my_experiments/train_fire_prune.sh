#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd ..
python train.py --epochs 120 --weights weights/yolov5l.pt --cfg models/yolov5l.yaml \
--data data/fire.yaml --batch-size 16 --workers 8 --device 1 --use-pruning --layer-rate 0.8 \
--layer-gap 0,320,3 --skip-downsample
cd -
