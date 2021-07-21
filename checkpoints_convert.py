import sys
import argparse
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

import torch
from models.yolol import Model

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='./weights/yolov5l.pt')
args = parser.parse_args()
weights_path = Path(args.weights)

model = torch.load(weights_path)
model_src = model['model']
model_src_dict = model_src.state_dict()
print('model_src_ckpt:{}'.format(model.keys()))
model_dst = Model(ch=3, nc=80)
model_dst_dict = model_dst.state_dict()
model_dst.load_state_dict(model_dst_dict)
model_dst_keys_list = list(model_dst_dict.keys())

# 两个模型的权重名字打印出来
for m1, m2 in zip(model_src_dict, model_dst_dict):
    print(m1, ' || ', m2)

for ind, value in enumerate(model_src_dict.values()):
    model_dst_dict[model_dst_keys_list[ind]] = value

ckpt = {
    'epoch': -1,
    'best_fitness': None,
    'training_results':None,
    'model':model_dst,
    'optimizer':None
}
torch.save(ckpt, weights_path.parents[0] / (weights_path.stem + '_new' + weights_path.suffix))


