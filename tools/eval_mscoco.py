import os, sys
sys.path.append(os.getcwd())
from api import *
import argparse
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import encode

### cfg ###
getting_results = True
num_datas = -500
eval_type = 'segm' # bbox or segm
########### 

parser = argparse.ArgumentParser(description='Pytorch Object Detection Train')
parser.add_argument(
    '--cfg',
    help='path to config file'
)
parser.add_argument("--len", type=int, default=-1)
parser.add_argument("--update", type=int, default=1)
parser.add_argument("--task", type=str, default='segm')
args = parser.parse_args()
cfg_file = args.cfg
num_datas = args.len
if args.update == 0: 
    getting_results = False
eval_type = args.task

cfg = load_cfg(cfg_file)
mode = 'EVAL'
res_file = cfg['weight_file'].split('.')[0] + '_mscoco_res.json'

prepare_device(cfg, mode)
detector = prepare_detector(cfg, mode)
dataset = prepare_dataset(cfg, detector, mode)
if num_datas>0:
    dataset.ids = dataset.ids[:num_datas]
inferencer = Inferencer(cfg, detector, dataset, mode)

# getting results
if getting_results:
    print('getting results ...')
    results = []
    for idx in range(len(dataset.ids)):
        img_name = dataset.coco.loadImgs(dataset.ids[idx])[0]['file_name']
        img = Image.open(os.path.join(dataset.root_img, img_name))
        pred = inferencer.pred(img)
        if pred['box'].shape[0] > 0:
            ymin, xmin, ymax, xmax = pred['box'].split([1, 1, 1, 1], dim=1)
            h, w = ymax - ymin + 1, xmax - xmin + 1
            pred_box = torch.cat([xmin, ymin, w, h], dim=1)
            if dataset.task == 'segm':
                for box_id in range(pred_box.shape[0]):
                    score = float(pred['score'][box_id])
                    label = int(dataset.index_to_coco[int(pred['class'][box_id])])
                    box = pred_box[box_id, :]
                    rle = encode(np.asfortranarray(pred['mask'][box_id].numpy().astype(np.uint8)))
                    rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings
                    image_result = {
                        'image_id'    : dataset.ids[idx],
                        'category_id' : label,
                        'score'       : score,
                        'bbox'        : box.tolist(),
                        'segmentation': rle,
                    }
                    results.append(image_result)
            elif dataset.task == 'bbox': 
                for box_id in range(pred_box.shape[0]):
                    score = float(pred['score'][box_id])
                    label = int(dataset.index_to_coco[int(pred['class'][box_id])])
                    box = pred_box[box_id, :]
                    image_result = {
                        'image_id'    : dataset.ids[idx],
                        'category_id' : label,
                        'score'       : score,
                        'bbox'        : box.tolist(),
                    }
                    results.append(image_result)
            else:
                raise NotImplementedError
        print('step:%d/%d' % (idx, len(dataset.ids)), end='\r')
    json.dump(results, open(res_file, 'w'), indent=4)


# evaluating
print('evaluating...')
coco_pred = dataset.coco.loadRes(res_file)
if dataset.task == 'segm':
    coco_eval = COCOeval(dataset.coco, coco_pred, eval_type)
else:
    coco_eval = COCOeval(dataset.coco, coco_pred, 'bbox')
coco_eval.params.imgIds = dataset.ids
coco_eval.evaluate()
coco_eval.accumulate()
print('iters:', int(detector.trained_log[0]), 
        ' epoches:', int(detector.trained_log[1]))
coco_eval.summarize()
