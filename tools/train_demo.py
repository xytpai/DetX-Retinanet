import os, sys
sys.path.append(os.getcwd())
from api import *
import argparse

parser = argparse.ArgumentParser(description='Pytorch Object Detection Train')
parser.add_argument(
    '--cfg',
    help='path to config file'
)
args = parser.parse_args()
cfg_file = args.cfg

cfg = load_cfg(cfg_file)
mode = 'TRAIN'

prepare_device(cfg, mode)
detector = prepare_detector(cfg, mode)
dataset = prepare_dataset(cfg, detector, mode)
loader = prepare_loader(cfg, dataset, mode)
opt = prepare_optimizer(cfg, detector, mode)
trainer = Trainer(cfg, detector, dataset, loader, opt)

while True:
    if trainer.step_epoch(save_last=True):
        break
print('Schedule finished!')
