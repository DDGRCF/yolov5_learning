import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None
logger = logging.getLogger(__name__)

class Model(nn.Module):
    
    def __init__(self, ch=3, nc=80, anchor=None):
        super(Model, self).__init__()
        self.backbone = Backbone()
        self.neck = Neck() 
        self.detect = Detect()

        self.init_weights()
    
    def init_weights(self):
        pass


class Backbone(nn.Module):
    
    def __init__(self, inchs=3):
        super(Backbone, self).__init__()
        self.focus = Focus(inchs, 64, 3)

        self.conv1 = Conv(64, 128, 3, 2)
        self.csp1 = C3(128, 128, 3)

        self.conv2 = Conv(128, 256, 3, 2)
        self.csp2 = C3(256, 256, 9)

        self.conv3 = Conv(256, 512, 3, 2)
        self.csp3 = C3(512, 512, 9)

        self.conv4 = Conv(512, 1024, 3, 2)
        self.spp = SPP(1024, 1024, [5, 9, 13])
        self.csp4 = C3(1024, 1024, 3, False)
    
    def forward(self, x):

        x_0 = self.focus(x)
        x_1 = self.conv1(x)
        x_2 = self.csp1(x)
        x_3 = self.conv2(x)
        x_4 = self.csp2(x)
        x_5 = self.conv3(x)
        x_6 = self.csp3(x)
        x_7 = self.conv4(x)
        x_8 = self.spp(x)
        x_9 = self.csp4(x)
    
        return x

class Neck(nn.Model):

    def __init__(self, inchs):
        super(Neck, self).__init__()
        self.conv1 = Conv(1024, 512, 1, 1)
        self.up1 = nn.Upsample(None, 2, 'nearest')
        self.concat1 = Concat(1) 
        self.csp1 = C3(1024, 512, 3, False)

        self.conv2 = Conv(512, 256, 1, 1)
        self.up2 = nn.Upsample(None, 2, 'nearest')

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
