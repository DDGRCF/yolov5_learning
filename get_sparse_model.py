import torch
import argparse
import logging
import yaml
import numpy as np
import os

from pathlib import Path
from utils.prune_utils import *
from utils.general import set_logging

set_logging()
logger = logging.getLogger(__name__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(opt):
    weights, data, device = opt.weights, opt.data, opt.device
    device = torch.device('cuda', int(device))
    weights = Path(weights)
    with open(data, 'r') as f:
        data_dict = yaml.safe_load(f)
    ckpt = torch.load(weights, map_location=device)
    model = ckpt['model'].float()
    skip_list = ckpt['remain_layers']
    bn_weights, highest_thre  = gather_bn_weights(model, skip_list)
    sorted_bn = torch.sort(bn_weights)[0]
    percent_limit = torch.sum(sorted_bn < highest_thre).item() / len(bn_weights)
    logger.info(f"Threshold should be less than {highest_thre}\nThe ratio will be {percent_limit}")
    model_eval(model, data_dict)
    model = mask_bn(model, skip_list, highest_thre)
    model_eval(model, data_dict)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default='weights/yolov5l.pt')
    parser.add_argument("--data", type=str, default='data/fire.yaml')
    parser.add_argument("--device", type=str, default='0')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = get_opt()
    main(opt)
