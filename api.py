import os, sys
import numpy as np
import torch
import time
import yaml
import random
import torchvision.transforms as transforms
import torch.nn.functional as F 
from datasets.utils import *


def load_cfg(cfg_file):
    cfg = None
    weight_file = None
    with open(cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        weight_file = 'weights/' + os.path.split(cfg_file)[1].split('.')[0] + '.pkl'
        cfg['weight_file'] = weight_file
    return cfg


def prepare_device(cfg, mode):
    if mode == 'TRAIN':
        torch.cuda.set_device(cfg['TRAIN']['DEVICES'][0])
        seed = cfg['TRAIN']['SEED']
        if seed >= 0:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.set_device(cfg[mode]['DEVICE'])


def prepare_detector(cfg, mode):
    dt = __import__('detectors.'+cfg['DETECTOR']['NAME'], 
                        fromlist=(cfg['DETECTOR']['NAME'],))
    detector = dt.Detector(cfg, mode=mode)
    if mode == 'TRAIN':
        if cfg['TRAIN']['LOAD']:
            detector.load_state_dict(torch.load(cfg['weight_file'], map_location='cpu'))
        detector = torch.nn.DataParallel(detector, device_ids=cfg['TRAIN']['DEVICES'])
        detector = detector.cuda(cfg['TRAIN']['DEVICES'][0])
        detector.train()
    else: 
        detector.load_state_dict(torch.load(cfg['weight_file'], map_location='cpu'))
        detector = detector.cuda(cfg[mode]['DEVICE'])
        detector.eval()
    return detector


def prepare_dataset(cfg, detector, mode):
    ds = __import__('datasets.'+cfg['DATASET']['NAME'], 
                        fromlist=(cfg['DATASET']['NAME'],))
    if mode == 'TRAIN':
        dataset = ds.Dataset(cfg['DATASET']['ROOT_TRAIN'], cfg['DATASET']['JSON_TRAIN'], 
                        cfg['TRAIN']['SIZE'], True, cfg['TRAIN']['TRANSFER_P'], 
                        cfg['TRAIN']['TRANSFER_MIN'])
    else:
        dataset = ds.Dataset(cfg['DATASET']['ROOT_'+mode], cfg['DATASET']['JSON_'+mode], 
                        cfg[mode]['SIZE'], True, 0.0, 1.0)
    return dataset


def prepare_loader(cfg, dataset, mode):
    assert mode == 'TRAIN'
    loader = data.DataLoader(dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], 
                        shuffle=True, num_workers=cfg['TRAIN']['NUM_WORKERS'], 
                        collate_fn=dataset.collate_fn)
    return loader


def prepare_optimizer(cfg, detector, mode):
    assert mode == 'TRAIN'
    lr_base = cfg['TRAIN']['LR_BASE']
    params = []
    for key, value in detector.named_parameters():
        if not value.requires_grad:
            continue
        _lr = lr_base
        _weight_decay = cfg['TRAIN']['WEIGHT_DECAY']
        if "bias" in key:
            _lr = lr_base * 2
            _weight_decay = 0
        params += [{"params": [value], "lr": _lr, "weight_decay": _weight_decay}]
    opt = torch.optim.SGD(params, lr=_lr, momentum=cfg['TRAIN']['MOMENTUM'])
    return opt


class Trainer(object):
    def __init__(self, cfg, detector, dataset, loader, opt):
        self.cfg = cfg
        self.detector = detector
        self.detector.train()
        self.dataset = dataset
        self.task = dataset.task
        self.loader = loader
        self.opt = opt
        self.step = int(self.detector.module.trained_log[0])
        self.epoch = int(self.detector.module.trained_log[1])
        # lr
        self.grad_clip = cfg['TRAIN']['GRAD_CLIP']
        self.lr_base = cfg['TRAIN']['LR_BASE']
        self.lr_gamma = cfg['TRAIN']['LR_GAMMA']
        self.lr_schedule = cfg['TRAIN']['LR_SCHEDULE']
        self.warmup_iters = cfg['TRAIN']['WARMUP_ITER']
        self.warmup_factor = 1.0/3.0  
        self.device = cfg['TRAIN']['DEVICES']
        self.save = cfg['TRAIN']['SAVE']
        
    def step_epoch(self, save_last=False):
        # freeze
        self.detector.module.backbone.freeze_stages(int(self.cfg['TRAIN']['FREEZE_STAGES']))
        if self.cfg['TRAIN']['FREEZE_BN']:
            self.detector.module.backbone.freeze_bn()
        
        if self.epoch >= self.cfg['TRAIN']['NUM_EPOCH']:
            if save_last:
                self.detector.module.trained_log[0] = self.step
                self.detector.module.trained_log[1] = self.epoch
                torch.save(self.detector.module.state_dict(), self.cfg['weight_file'])
            return True
        
        for i, data in enumerate(self.loader):
            # lr function
            lr = self.lr_base
            if self.step < self.warmup_iters:
                alpha = float(self.step) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1.0 - alpha) + alpha
                lr = lr*warmup_factor 
            else:
                for j in range(len(self.lr_schedule)):
                    if self.step < self.lr_schedule[j]:
                        break
                    lr *= self.lr_gamma
            for param_group in self.opt.param_groups:
                param_group['lr'] = lr
            # #########
            if i == 0: batch_size = int(data['imgs'].shape[0])
            time_start = time.time()
            self.opt.zero_grad()
            if self.task == 'segm':            
                loss = self.detector(data['imgs'], data['locations'], data['labels'], 
                                        data['boxes'], data['masks']).mean()
            elif self.task == 'bbox':
                loss = self.detector(data['imgs'], data['locations'], data['labels'], 
                                        data['boxes']).mean()
            else: 
                raise NotImplementedError
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.detector.parameters(), self.grad_clip)
            self.opt.step()
            maxmem = int(torch.cuda.max_memory_allocated(device=\
                self.device[0]) / 1024 / 1024)
            time_end = time.time()
            totaltime = int((time_end - time_start) * 1000)
            print('total_step:%d: epoch:%d, step:%d/%d, loss:%f, maxMem:%dMB, time:%dms, lr:%f' % \
                (self.step, self.epoch, i*batch_size, len(self.dataset), loss, maxmem, totaltime, lr))
            self.step += 1
        self.epoch += 1
        if self.save:
            self.detector.module.trained_log[0] = self.step
            self.detector.module.trained_log[1] = self.epoch
            torch.save(self.detector.module.state_dict(), self.cfg['weight_file'])
        return False


class Inferencer(object):
    def __init__(self, cfg, detector, dataset, mode):
        self.cfg = cfg
        self.detector = detector
        self.detector.eval()
        self.mode = mode
        self.normalizer = dataset.normalizer
        self.task = dataset.task
        assert self.mode != 'TRAIN'

    def pred(self, img_pil):
        with torch.no_grad():
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            img_pil, location, _, _ = to_square(img_pil, 
                    self.cfg[self.mode]['SIZE'], mode=self.mode)
            img = transforms.ToTensor()(img_pil)
            img = self.normalizer(img)
            img = img.cuda().unsqueeze(0)
            if self.task == 'segm':
                pred_cls_i, pred_cls_p, pred_reg, pred_mask = self.detector(img, location)
                return {'class':pred_cls_i.cpu(), 'score':pred_cls_p.cpu(), 
                            'box':pred_reg.cpu(), 'mask':pred_mask.cpu()}
            elif self.task == 'bbox': 
                pred_cls_i, pred_cls_p, pred_reg = self.detector(img, location)
                return {'class':pred_cls_i.cpu(), 'score':pred_cls_p.cpu(), 
                            'box':pred_reg.cpu()}
