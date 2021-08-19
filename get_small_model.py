import sys
from typing import OrderedDict
import torch
import numpy as np
import argparse
import logging
import yaml
import json
from models.yolo import Model as Model
from models.yolov5l_pruning import Model as Model_Pruning
from pathlib import Path
from copy import deepcopy
from utils.general import set_logging
from tqdm import tqdm
from utils.prune_utils import model_compare, model_eval
from utils.torch_utils import select_device
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
set_logging()
logger = logging.getLogger(__name__)

def main(opt):
    setup_seed(20)
    weights, cfg, data, device = opt.weights, opt.cfg, opt.data, opt.device
    # Load the model
    device = select_device(device)
    weights = Path(weights)
    pruning_weights = weights.parents[0] / ('pruning_' + weights.name)
    pruning_cfg = weights.parents[0] / 'pruning_cfg.json'
    cfg = Path(cfg)
    assert weights.match('*.pt'), 'the file must be the type of *.pt '
    ckpt = torch.load(weights, map_location=lambda storage, loc: storage) # 将权重从gpu上加载进cpu中
    assert cfg.match('*.yaml'), 'the file must be the type of *.yaml'
    with open(cfg) as f:
        cfg = yaml.safe_load(f)
    data = Path(data)
    with open(data) as f:
        data_dict = yaml.safe_load(f)

    # Load the model configuration
    model = ckpt['model'].float()
    cfg['nc'] = model.yaml.get('nc', 1)

    input_from = OrderedDict()
    shortcut = OrderedDict()

    # get orginal model configuration
    for i, m in enumerate(cfg['backbone'] + cfg['head']):
        input_from[i] = m[0]
        shortcut[i] = m[3][-1]

    # Pruning the params
    first_layer = 0
    last_layer = 641
    inter_layer = 6
    layer_index = [i for i in range(first_layer, last_layer + 1, inter_layer)]
    channel_index = OrderedDict()
    state_dict_ = OrderedDict()
    keep_mask = []
    state_dict = model.state_dict()

    # prune the output channels params
    len_pbar = len(state_dict)
    pbar = tqdm(enumerate(state_dict.items()), total=len_pbar)

    for i, (n, param) in pbar:
        m_index = int(n.split('.')[1])
        if i in layer_index: # 获得保留权重的通道和去除权重通道的index
            if shortcut[m_index] == True and '.cv1.' in n and '.m.' not in n:  
                keep_index = torch.LongTensor(range(param.shape[0]))
            else:
                keep_index = torch.nonzero(torch.sum(param.data, dim=(1, 2, 3)) != 0).view(-1) 
            state_dict_[n] = torch.index_select(param, 0, keep_index)

            keep_mask.append((param.shape, keep_index))
            if m_index in channel_index:
                channel_index[m_index] += [(param.shape, keep_index)]
            else:
                channel_index[m_index] = [(param.shape, keep_index)]
 
        elif (i not in layer_index) and (first_layer < i < last_layer + 1):
            if len(param.shape) != 0:
                state_dict_[n] = torch.index_select(param, 0, keep_mask[-1][-1])
            else:
                state_dict_[n] = param.clone()
        else:
            state_dict_[n] = param.clone()
    logger.info('the output channels pruning is completed!!!')
    # del state_dict_strip
    # prune the input channels
    len_pbar = len(state_dict_)
    pbar = tqdm(enumerate(state_dict_.items()), total=len_pbar)
    index_ = -1
    for i, (n, param) in pbar:
        m_index = int(n.split('.')[1])
        if i in layer_index:
            index_ += 1
            if index_ == 0:
                continue
            if  '.m.' not in n:
                if '.cv1.' in n:
                    f = input_from[m_index - 1]
                    if isinstance(f, list):
                        k1 = keep_mask[index_ - 1][-1]
                        k2 = channel_index[f[-1]][-1][-1] + keep_mask[index_ - 1][0][0]
                        k = torch.cat((k1, k2), dim=0)
                    else:
                        k = keep_mask[index_ - 1][-1]
                    state_dict_[n] = torch.index_select(param, 1, k)
                    continue
                elif ('cv2' in n) and (m_index == 8):
                    k = torch.cat([(keep_mask[index_ - 1][-1] + d * keep_mask[index_ - 1][0][0]) for d in range(4)])
                    state_dict_[n] = torch.index_select(param, 1, k)
                    continue
                elif ('cv2' in n) and (m_index != 8):
                    f = input_from[m_index - 1]
                    if isinstance(f, list):
                        k1 = keep_mask[index_ - 2][-1]
                        k2 = channel_index[f[-1]][-1][-1] + keep_mask[index_ - 2][0][0]
                        k = torch.cat((k1, k2), dim=0)
                    else:
                        k = keep_mask[index_ - 2][-1]
                    state_dict_[n] = torch.index_select(param, 1, k)
                    continue
                elif 'cv3' in n:
                    if shortcut[m_index] == True:
                        k1 = torch.LongTensor(range(keep_mask[index_ - 1][0][0]))
                    else:
                        k1 = keep_mask[index_ - 1][-1]
                    k2 = channel_index[m_index][1][-1] + keep_mask[index_ - 1][0][0]
                    k = torch.cat((k1, k2), dim=0)
                    state_dict_[n] = torch.index_select(param, 1, k)
                    continue
            if shortcut[m_index] == True and '.cv1.' in n:
               k = torch.LongTensor(range(keep_mask[index_ - (2 if '.0.' in n else 1)][0][0]))
            else:
                k = keep_mask[index_ - (2 if '.0.' in n else 1)][-1]
            state_dict_[n] = torch.index_select(param, 1, k)
        elif i > last_layer:
            if 'anchor' not in n:
                f = input_from[m_index]
                if 'weight' in n:
                    k = channel_index[f[int((i - last_layer - 3) / 2)]][-1][-1]
                    state_dict_[n] = torch.index_select(param, 1, k)

    # get pruning cfg information
    logger.info('the input channels pruning is completed!!!')
    channel_index = OrderedDict()
    index_ = -1
    for i, (n, param) in enumerate(state_dict_.items()):
        m_index = int(n.split('.')[1])
        shape = list(param.shape[: 2])
        if i in layer_index:
            index_ += 1
            if m_index in channel_index:
                channel_index[m_index] += [(shape, keep_mask[index_][-1].tolist())]
            else:
                channel_index[m_index] = [(shape, keep_mask[index_][-1].tolist())]
        elif i > last_layer:
            if 'anchor' not in n:
                if 'weight' in n:
                    if m_index in channel_index:
                        channel_index[m_index] += [shape[1]]
                    else:
                        channel_index[m_index] = [shape[1]]

    # 将裁剪的配置保持成json文件
    with open(pruning_cfg, 'w') as f:
        f.write(json.dumps(channel_index))
    
    # 初始化小模型
    model_pruning = Model_Pruning(cfg, nc=1, channel_index=channel_index)

    # 小模型加载参数
    new_state_dict = OrderedDict()
    for i, (n1, n2) in enumerate(zip(state_dict_.items(), model_pruning.state_dict().items())):
        assert n1[-1].shape == n2[-1].shape, 'there are errors in state_dict_'
        new_state_dict[n2[0]] = state_dict_[n1[0]]

    model_pruning.load_state_dict(new_state_dict, strict=True)

    # evaluate the model
    model_compare(model, model_pruning)
    
    model_eval(model, data_dict, device)
    model_eval(model_pruning, data_dict, device)
    # save model
    ckpt['model'] = deepcopy(model_pruning.half())
    ckpt['best_fitness'] = 0
    ckpt['epoch'] = -1
    torch.save(ckpt, pruning_weights)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5l.pt')
    parser.add_argument('--cfg', type=str, default='models/yolov5l.yaml')
    parser.add_argument('--data', type=str, default='data/fire.yaml')
    parser.add_argument('--device', type=str, default='0')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = get_opt()
    main(opt)