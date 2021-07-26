import torch
import numpy as np
import logging
from utils.general import colorstr
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
        self.model = model
        self.mask_index = []
        self.opt = opt if opt is not None else kwargs

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()
            self.model_length[index] = np.prod(self.model_size[index])

    def init_rate(self, layer_rate):
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
        for key in range(self.opt.layer_begin, self.opt.layer_end, self.opt.layer_inter):
            self.compress_rate[key] = layer_rate
        last_index = None
        skip_list = []
        self.mask_index = [x for x in range(0, last_index, 3)]
        if self.opt.skip_downsample == 1:
            for x in skip_list:
                self.compress_rate[x] = 1
                self.mask_index.remove(x)
        prefix = colorstr('mask_index:')
        logger.info('{}{}'.format(prefix, self.mask_index))
    
    def init_mask(self, layer_rate):
        self.init_rate(layer_rate)
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index], 
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                if self.cuda:
                    self.mat[index] = self.mat[index].to(self.device)
        logging.info('mask Ready')

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        logging.info('mask Done')
        
    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate)) 
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
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

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            if index in [x for x in range(self.opt.layer_begin, self.opt.layer_end, self.opt.layer_inter)]:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()
                prefix = colorstr('layer:')
                logger.info('{}{}, number of nonzero weight is {}, zero is {}'.format(prefix, 
                            index, np.count_nonzero(b), len(b) - np.count_nonzero(b)))
