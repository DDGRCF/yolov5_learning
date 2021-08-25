#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5

cd ..
python train.py --epochs 220 --weights weights/yolov5l.pt --cfg models/yolov5l.yaml --data data/fire.yaml --batch-size 16 --device 2 --use-pruning --skip-list 0 3 --pruning-method Network_Slimming --s 0.01
cd -

