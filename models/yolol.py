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
    
    def __init__(self, ch=3, nc=80, inplace=True):
        super(Model, self).__init__()
        self.anchors = [[10,13, 16,30, 33,23], 
                        [30,61, 62,45, 59,119], 
                        [116,90, 156,198, 373,326]]
        self.backbone = Backbone(ch)
        self.neck = Neck() 
        self.detect = Detect(nc, self.anchors, (256, 512, 1024))
        self.names = [str(i) for i in range(nc)]  # default names
        self.inplace = inplace
        if isinstance(self.detect, Detect):
            s = 256
            self.detect.inplace = self.inplace
            self.detect.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
            check_anchor_order(self.detect)
            self.stride = self.detect.stride
            self._initialize_biases()

        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, visualize=False):
        if augment:
            return self.forward_augment(x)
        return self.forward_once(x)

    def forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x):

        y, f1 = self.backbone(x)
        f2 = self.neck(y, f1)
        out = self.detect(f2)
        
        return out

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _print_biases(self):
        m = self.detect  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        split = ['self.backbone', 'self.neck', 'self.detect']
        for s in split:
            module = eval(f'{s}.modules()')
            for m in module:
                if type(m) is Conv and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def autoshape(self):  # add AutoShape module
        logger.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    
    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


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

        x_1 = self.conv1(x_0)
        x_2 = self.csp1(x_1)

        x_3 = self.conv2(x_2)
        x_4 = self.csp2(x_3)

        x_5 = self.conv3(x_4)
        x_6 = self.csp3(x_5)

        x_7 = self.conv4(x_6)
        x_8 = self.spp(x_7)
        x_9 = self.csp4(x_8)

        out_features = {'x_4':x_4, 'x_6':x_6}

        return x_9, out_features

class Neck(nn.Module):

    def __init__(self):
        super(Neck, self).__init__()
        self.conv1 = Conv(1024, 512, 1, 1)
        self.up1 = nn.Upsample(None, 2, 'nearest')
        self.concat1 = Concat(1) 
        self.csp1 = C3(1024, 512, 3, False)

        self.conv2 = Conv(512, 256, 1, 1)
        self.up2 = nn.Upsample(None, 2, 'nearest')
        self.concat2 = Concat(1)
        self.csp2 = C3(512, 256, 3, False)

        self.conv3 = Conv(256, 256, 3, 2)
        self.concat3 = Concat(1)
        self.csp3 = C3(512, 512, 3, False)

        self.conv4 = Conv(512, 512, 3, 2)
        self.concat4 = Concat(1)
        self.csp4 = C3(1024, 1024, 3, False)

    def forward(self, x, cat_features):
        x_10 = self.conv1(x)
        x_11 = self.up1(x_10)
        x_12 = self.concat1([x_11, cat_features['x_6']])
        x_13 = self.csp1(x_12)

        x_14 = self.conv2(x_13)
        x_15 = self.up2(x_14)
        x_16 = self.concat2([x_15, cat_features['x_4']])
        x_17 = self.csp2(x_16)

        x_18 = self.conv3(x_17)
        x_19 = self.concat3([x_18, x_14])
        x_20 = self.csp3(x_19)
        
        x_21 = self.conv4(x_20)
        x_22 = self.concat4([x_21, x_10])
        x_23 = self.csp4(x_22)

        return [x_17, x_20, x_23]

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
