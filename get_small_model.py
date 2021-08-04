import sys
import torch
import numpy as np
import argparse
import logging
from models.yolo import Model
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parent[0].as_posix())

logger = logging.getLogger(__name__)

def main(opt):
    weights = opt.weights
    weights = Path(weights)
    assert weights.match('*.pt'), 'the file must be the type of *.pt '
    ckpt = torch.load(weights, map_location=lambda storage, loc: storage) # 将权重从gpu上加载进cpu中
    model = ckpt['model']
    for i, (n, p) in enumerate(model.named_parameters()):
        print(i, n, p)
    

def get_opt():
    pass

if __name__ == '__name__':
    opt = get_opt()
    main(opt)