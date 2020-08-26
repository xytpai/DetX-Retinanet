import os, sys
sys.path.append(os.getcwd())
from api import *
import argparse
from PIL import Image
from datasets.utils import *

demo_dir = 'demo_images'
parser = argparse.ArgumentParser(description='Pytorch Object Detection Train')
parser.add_argument(
    '--cfg',
    help='path to config file'
)
args = parser.parse_args()
cfg_file = args.cfg

cfg = load_cfg(cfg_file)
mode = 'TEST'

prepare_device(cfg, mode)
detector = prepare_detector(cfg, mode)
dataset = prepare_dataset(cfg, detector, mode)
inferencer = Inferencer(cfg, detector, dataset, mode)

for filename in os.listdir(demo_dir):
    if filename.endswith('jpg'):
        if filename[:5] == 'pred_': 
            continue
        img = Image.open(os.path.join(demo_dir, filename))
        pred = inferencer.pred(img)
        name = demo_dir + '/pred_' + filename.split('.')[0]+'.jpg'
        if dataset.task == 'segm':
            show_instance(img, pred['box'], pred['class'], pred['mask'], 
                scores=pred['score'], name_table=dataset.name_table, 
                file_name=name)
        elif dataset.task == 'bbox':
            show_instance(img, pred['box'], pred['class'], 
                scores=pred['score'], name_table=dataset.name_table, 
                file_name=name)
