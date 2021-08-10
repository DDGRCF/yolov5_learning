from typing import OrderedDict
import torch
import numpy as np
import logging
from utils.general import colorstr
from utils.torch_utils import is_parallel
logger = logging.getLogger(__name__)

class Mask:
    def __init__(self, 
                 model, 
                 device, 
                 opt=None, 
                 **kwargs):
        self.cuda = device.type != 'cpu' 
        self.device = device
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model.module if is_parallel(model) else model
        self.mask_index = []
        self.opt = opt if opt is not None else kwargs
        self.layer_begin, self.layer_end, self.layer_inter = [
            int(i.strip(' ')) for i in opt.layer_gap.split(',')
        ]
    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()
            self.model_length[index] = np.prod(self.model_size[index])

    def init_rate(self, layer_rate):
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
        for key in range(self.layer_begin, self.layer_end, self.layer_inter):
            self.compress_rate[key] = layer_rate
        last_index = 321
        skip_list = [0, 3] #[x for x in range(0, last_index, 3) if x not in range(0, 201, 3)]
        # [0,3, 
        # 6,15,21,27,
        # 36,45,51,57,63,69,75,81,87,93,
        # 102,111,117,123,129,135,141,147,153,159,
        # 174,183,189,195]

        self.mask_index = [x for x in range(0, last_index, 3)]
        if self.opt.skip_downsample:
            for x in skip_list:
                self.compress_rate[x] = 1
                self.mask_index.remove(x)

    def init_mask(self, layer_rate, print_info=True):
        self.init_rate(layer_rate)
        if print_info:
            prefix = colorstr('mask_index:')
            logging.info('{}\n{}'.format(prefix, self.mask_index))

        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index], 
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                if self.cuda:
                    self.mat[index] = self.mat[index].to(self.device)
        logging.info('Mask Ready...')

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        logging.info('Mask Done...')
        
    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate)) 
            weight_vec = weight_torch.view(weight_torch.size()[0], -1) # view不是inplace，重新复制了一份
            norm2 = torch.norm(weight_vec, 2, 1) 
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            kernel_length = np.prod(weight_torch.size()[1: ])
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0
            return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x) 
        return x

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        return weight_np

    def if_zero(self, epoch=None, save_file=None):
        prefix_print = True
        for index, item in enumerate(self.model.parameters()):
            if index in [x for x in range(self.layer_begin, self.layer_end, self.layer_inter)]:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()
                if save_file is not None:
                    assert save_file.match('*.txt'), 'the prune save must be txt'
                    with open(save_file, 'a') as fw:
                        if epoch is not None and prefix_print:
                            prefix = '>' * 20 + f' epoch:{epoch} ' + '<' * 20 + '\n'
                            fw.write(prefix)
                            prefix_print = False
                        fw.write('layer:{}, number of nonzero weight is {}, zero is {}\n'.format(index, 
                                  np.count_nonzero(b), len(b) - np.count_nonzero(b)))
                else:
                    prefix = colorstr('layer:')
                    logging.info('{}{}, number of nonzero weight is {}, zero is {}'.format(prefix, 
                                index, np.count_nonzero(b), len(b) - np.count_nonzero(b)))


def get_pruning_cfg(cfg):
    new_cfg = OrderedDict()
    for k, v in cfg.items():
        for cfg in v:
            if isinstance(cfg, list):
                cfg = tuple(cfg)
            if isinstance(k, str):
                k = int(k)
            if k in new_cfg:
                new_cfg[k] += [cfg]
            else:
                new_cfg[k] = [cfg]
    logging.info('the pruning configuration convertion complete!')
    return new_cfg

                
