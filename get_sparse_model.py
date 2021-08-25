import torch
import argparse
import logging
import yaml
import numpy as np
import os

from pathlib import Path
from utils.prune_utils import *
from utils.general import set_logging
from utils.torch_utils import select_device
from get_small_model import pruning

set_logging()
logger = logging.getLogger(__name__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
 
@torch.no_grad()
def main(opt):
    weights, data, device = opt.weights, opt.data, opt.device
    device = select_device(device)
    weights = Path(weights)
    pruning_path_dir = weights.parents[0]
    with open(data, 'r') as f:
        data_dict = yaml.safe_load(f)
    ckpt = torch.load(weights, map_location=lambda storage, loc: storage)
    model = ckpt['model'].float()
    skip_list = ckpt['skip_list']
    bn_weights, highest_thre  = gather_bn_weights(model, skip_list)
    sorted_bn = torch.sort(bn_weights)[0]
    percent_limit = torch.sum(sorted_bn < highest_thre).item() / len(sorted_bn)
    logger.info(f"Threshold should be less than {highest_thre}\nThe ratio will be {percent_limit}")
    model_eval(model, data_dict, device)
    get_sparse_model(model, skip_list, opt.ratio, sorted_bn)
    pruning(model, device, opt.cfg, data_dict, pruning_path_dir)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default='weights/yolov5l.pt')
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument("--cfg", type=str, default='models/yolo5vl.yaml')
    parser.add_argument("--data", type=str, default='data/fire.yaml')
    parser.add_argument("--ratio", type=float, default='0.7')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = get_opt()
    main(opt)
