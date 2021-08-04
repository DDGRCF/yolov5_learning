import argparse 
import logging
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import val
from copy import deepcopy
from torch.cuda import amp
from pathlib import Path
from tqdm import tqdm
from threading import Thread
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())
from utils.torch_utils import ModelEMA, select_device, de_parallel, intersect_dicts, ModelEMA
from utils.general import init_seeds, check_img_size, labels_to_class_weights, increment_path, set_logging
# from utils.google_utils import attempt_download
from utils.datasets import create_dataloader
from utils.loss import ComputeLoss
from utils.prune_utils import Mask
from models.experimental import attempt_load
from models.yolol import Model

set_logging()
logger = logging.getLogger()
RANK = int(os.getenv('RANK', -1))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5l_new.pt', help='the pretrained weights')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='the dataset configuration')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml')
    parser.add_argument('--project', type=str, default='runs/train', help='the train info dir')
    parser.add_argument('--epochs', type=int, default=100, help='the train epochs')
    parser.add_argument('--device', type=str, default='0, 1')
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8, help='train batch size')
    parser.add_argument('--workers', type=int, default=4, help='the workers')
    parser.add_argument('--resume', type=str, default='', help='empty is not resume')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, val] image sizes')
    parser.add_argument('--logging_fre', type=int, default=5, help='the info frequency to print')
    opt = parser.parse_args()
    project = Path(opt.project)
    opt.save_dir = str(increment_path(project / Path(opt.data).stem, exist_ok=True, mkdir=False))
    return opt

def train(hyp, opt, device):
    save_dir, epochs, batch_size, weights, data, workers, resume = \
    opt.save_dir, opt.epochs, opt.batch_size, opt.weights, opt.data, \
    opt.workers, opt.resume

    # Directory
    save_dir = Path(save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'

    # Hyperparameter
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)

    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    
    # Configure
    init_seeds(1+RANK)
    cuda = device.type != 'cpu'
    with open(data) as f:
        data_dict = yaml.safe_load(f)
    nc = int(data_dict['nc'])
    names = data_dict['names']
    
    # Model
    pretrained = weights.strip().endswith('pt')
    if pretrained:
        weights = Path(weights)
        assert weights.exists(), 'the pretrained weight dont exist'
        ckpt = torch.load(weights, map_location=device)
        exclude = ['anchor'] if not resume else []
        model = Model(ch=3, nc=nc).to(device)
        state_dict = ckpt['model'].float().state_dict()
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)
    data_root = Path(data_dict['path'])
    train_path = data_root / data_dict['train']
    val_path = data_root / data_dict['val']

    # Optimizer
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    optimizer = param_group_train(model, hyp)
    lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if pretrained:
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1 if ckpt['epoch'] is not None else 1

    del ckpt, state_dict
    # EMA
    ema = ModelEMA(model)
    # Datasets config
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.detect.nl
    imgsz, imgsz_val = [check_img_size(x, gs) for x in opt.img_size]

    # DP model
    if cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    dataloader = create_dataloader(train_path, imgsz, batch_size, gs, False, 
                                             hyp=hyp, augment=True, rect=False, workers=workers)[0]
    nb = len(dataloader)
    valloader = create_dataloader(val_path, imgsz_val, batch_size // 2, gs, False,
    hyp=hyp, rect=True, workers=workers, pad=0.5)[0]
    
    # model.half().float()

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.names = names

    # start training_results
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model, prune=True)
    last_opt_step = -1
    map = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        pbar = tqdm(enumerate(dataloader), total=nb)
        optimizer.zero_grad()
        mloss = torch.zeros(4, device=device)
        logger.info(('\n' + '%10s' * 7) % ('Epoch', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float()

            # Warmup
            if ni <= nw:
                xi =  [0, nw]
                accumulate = max(1, np.interp(ni, xi, [0, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            with amp.autocast(enabled=cuda):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))
                scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
                last_opt_step = ni
            mloss = (mloss * i + loss_items) / (i + 1)
            s = ('%10s' + '%10.4g' * 6) % (f'{epoch}/{epochs - 1}', *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)
        # lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()
        ckpt = {
            'epoch': epoch,
            'model': deepcopy(de_parallel(model)).half(),
            'optimizer': optimizer.state_dict(),
            'ema':deepcopy(ema.ema).half(),
            'update':ema.updates,
            'optimizer':optimizer.state_dict()
        }
        torch.save(ckpt, last)
        for m in [last, best] if best.exists() else [last]:
            results, _, _ = val.run(data_dict, 
                                    batch_size=batch_size // 2, 
                                    imgsz=imgsz_val,
                                    model=attempt_load(m, map_location=device).half(),
                                    single_cls=False,
                                    dataloader=valloader,
                                    save_dir=save_dir,
                                    save_json=True,
                                    plots=False
                                    )
            map = results[3]


def param_group_train(model, hyp):
    pg0, pg1, pg2 = [], [], []
    for k,v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v, nn.Parameter):
            pg1.append(v.weight)
    
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    del pg0, pg1, pg2

    return optimizer

def main(opt):
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.resume:
        opt.weights = opt.resume
        logging.info('Resume from {}'.format(opt.resume))
    train(opt.hyp, opt, device)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)