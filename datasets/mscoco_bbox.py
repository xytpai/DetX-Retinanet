import torch
import torch.nn.functional as F
import numpy as np 
import os, math, random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import cv2
if __name__ != '__main__':
    from datasets.utils import *
else:
    from utils import *


class Dataset(data.Dataset):
    name_table = ['background', 
               'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, root_img, file_json, size, normalize, transfer_p, transfer_min):
        assert size%2 == 1
        self.root_img = root_img
        self.file_json = file_json
        self.size = size
        self.normalize = normalize
        self.transfer_p = transfer_p
        self.transfer_min = transfer_min
        # other
        self.task = 'bbox'
        self.normalizer = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        # get coco
        self.coco = COCO(file_json)
        self.index_to_coco = [i for i in range(len(self.name_table))]
        self.coco_to_index = {}
        for cate in self.coco.loadCats(self.coco.getCatIds()):
            name = cate['name']
            if name in self.name_table:
                index = self.name_table.index(name)
                self.index_to_coco[index] = cate['id']
                self.coco_to_index[cate['id']] = index
        # filter imgs
        self.ids = [] # [xxxx, xxxx, ...]
        for image_id in self.coco.getImgIds():
            img_info = self.coco.loadImgs(image_id)[0]
            if min(img_info['width'], img_info['height']) < 32:
                continue
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id, iscrowd=False))
            if len(anns) == 0:
                continue
            ena = False
            for ann in anns:
                if ann['category_id'] not in self.coco_to_index:
                    continue
                xmin, ymin, w, h = ann['bbox']
                if w < 1 or h < 1: 
                    continue
                ena = True
                break
            if ena: self.ids.append(image_id)
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        '''
        Return:
        img:      F(3, size, size)
        location: F(5)
        boxes:    F(n, 4)
        labels:   L(n)
        '''
        img_info = self.coco.loadImgs(self.ids[idx])[0]
        img_name = img_info['file_name']
        img = Image.open(os.path.join(self.root_img, img_name))
        if img.mode != 'RGB': 
            img = img.convert('RGB')
        boxes, labels = [], []
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=self.ids[idx], 
                                    iscrowd=False)) # keep iscrowd False
        for i, ann in enumerate(anns):
            if ann.get('ignore', False): continue
            xmin, ymin, w, h = ann['bbox']
            inter_w = max(0, min(xmin + w, img_info['width']) - max(xmin, 0))
            inter_h = max(0, min(ymin + h, img_info['height']) - max(ymin, 0))
            if inter_w * inter_h == 0: continue
            if ann['area'] <= 0 or w < 1 or h < 1: continue
            coco_id = ann['category_id']
            if coco_id not in self.coco_to_index: continue
            label = self.coco_to_index[coco_id]
            xmax, ymax = xmin + w - 1, ymin + h - 1
            boxes.append(torch.FloatTensor([ymin, xmin, ymax, xmax]))
            labels.append(torch.LongTensor([label]))
        if len(labels) > 0:
            boxes = torch.stack(boxes)
            labels = torch.cat(labels)
        else: # only bg
            boxes = torch.zeros(1, 4)
            labels = torch.zeros(1)
        boxes[:, :2].clamp_(min=0)
        boxes[:, 2].clamp_(max=float(img.size[1])-1)
        boxes[:, 3].clamp_(max=float(img.size[0])-1)
        if random.random() < 0.5: 
            img, boxes, _ = x_flip(img, boxes)
        img, location, boxes, _ = to_square(img, self.size, 
                                    self.transfer_p, self.transfer_min, boxes)
        img = transforms.ToTensor()(img)
        # normalize
        if self.normalize:
            img = self.normalizer(img)
        return img, location, boxes, labels

    def collate_fn(self, data):
        '''
        Return:
        imgs:      F(b, 3, size, size)
        locations: F(b, 5)
        boxes:     F(b, max_n, 4)
        labels:    L(b, max_n)            bg:0
        '''
        imgs, locations, boxes, labels = zip(*data)
        imgs = torch.stack(imgs)
        locations = torch.stack(locations)
        batch_num = len(imgs)
        max_n = 0
        for b in range(batch_num):
            if boxes[b].shape[0] > max_n: max_n = boxes[b].shape[0]
        boxes_t = torch.zeros(batch_num, max_n, 4)
        labels_t = torch.zeros(batch_num, max_n).long()
        for b in range(batch_num):
            boxes_t[b, :boxes[b].shape[0]] = boxes[b]
            labels_t[b, :boxes[b].shape[0]] = labels[b]
        return {'imgs':imgs, 'locations':locations, 
                    'boxes':boxes_t, 'labels':labels_t}


if __name__ == '__main__':
    root_img = 'D:\\dataset\\microsoft-coco\\val2017'
    file_json = 'D:\\dataset\\microsoft-coco\\instances_val2017.json'
    size = 641
    normalize = False
    transfer_p = 1.0
    transfer_min = 0.5
    batch_size = 4
    dataset = Dataset(root_img, file_json, size, normalize, 
                            transfer_p, transfer_min)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=0, collate_fn=dataset.collate_fn)
    for data in loader:
        imgs, locations, boxes, labels = data['imgs'], \
            data['locations'], data['boxes'], data['labels']
        print('imgs:', imgs.shape)
        print('locations:', locations.shape)
        print('boxes:', boxes.shape)
        print('labels:', labels.shape)
        b = random.randint(0, batch_size-1)
        show_instance(imgs[b], boxes[b], labels[b], name_table=dataset.name_table)
        break
