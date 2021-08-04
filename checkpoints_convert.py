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

model_ckpt = torch.load(weights_path)
model_src = model_ckpt['model']
model_src_dict = model_src.state_dict()

model_dst = Model(ch=3, nc=80)
model_dst_dict = model_dst.state_dict()

model_dst_keys_list = list(model_dst_dict.keys())

for i, (k, v) in enumerate(model_src.state_dict().items()):
    if 'num_batches_tracked' in k:
        print(f'{i}|{k}|{v}')
# for i, n in enumerate(zip(model_dst.named_parameters(), model_src.named_parameters())):
#     print(f'{i}|{n[1][0]}|{n[0][0]}|{n[1][1].shape}')

# for ind, value in enumerate(model_src_dict.values()):
#     model_dst_dict[model_dst_keys_list[ind]] = value
model_dst.load_state_dict(model_dst_dict)
ckpt = {
    'epoch': -1,
    'best_fitness': None,
    'training_results':None,
    'model':model_dst,
    'optimizer':None
}
torch.save(ckpt, weights_path.parents[0] / (weights_path.stem + '_new' + weights_path.suffix))


