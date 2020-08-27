import numpy as np
from PIL import Image, ImageDraw
import scipy.ndimage
import torch
import torch.utils.data as data
import warnings
import torchvision.transforms as transforms
import random
warnings.filterwarnings("ignore")


def filter_annotation(anno, class_id_set, height, width, hw_th=1, area_th=1):
    anno = [obj for obj in anno if not obj.get('ignore', False)]
    anno = [obj for obj in anno if obj['iscrowd'] == 0] # filter crowd annotations
    anno = [obj for obj in anno if obj['area'] >= area_th]
    anno = [obj for obj in anno if all(o >= hw_th for o in obj['bbox'][2:])]
    anno = [obj for obj in anno if obj['category_id'] in class_id_set]
    _anno = []
    for obj in anno:
        xmin, ymin, w, h = obj['bbox']
        inter_w = max(0, min(xmin + w, width) - max(xmin, 0))
        inter_h = max(0, min(ymin + h, height) - max(ymin, 0))
        if inter_w * inter_h > 0: _anno.append(obj)
    return _anno


def x_flip(img, boxes=None, masks=None):
    # return:
    # img:   PIL
    # boxes: arr(n, 4) or None : ymin, xmin, ymax, xmax
    # masks: arr(n, h, w) or None
    img = img.transpose(Image.FLIP_LEFT_RIGHT) 
    w = img.width
    if boxes is not None and boxes.shape[0] != 0:
        xmin = w - boxes[:, 3] - 1
        xmax = w - boxes[:, 1] - 1
        boxes[:, 1] = xmin
        boxes[:, 3] = xmax
    if masks is not None and masks.shape[0] != 0:
        masks = masks[:, :, ::-1]
    return img, boxes, masks


def to_square(img, size, 
                transfer_p=0.0, transfer_min=1.0, 
                boxes=None, masks=None, mode='TRAIN'):
    # size:  int
    # transfer_p: float (0~1)
    # transfer_min: float (0~1)
    # return:
    # img:   PIL(3, size, size)
    # location: arr(5) : ymin, xmin, ymax, xmax, scale_rate
    # boxes: arr(n, 4) or None : ymin, xmin, ymax, xmax
    # masks: ndarr(n, size, size) or None (hw)
    w, h = img.size
    size_min = min(w, h)
    size_max = max(w, h)
    _scale_rate = float(size) / size_max
    if mode == 'TRAIN' and random.random() < transfer_p:
        _scale_rate *= random.uniform(transfer_min, 1.0)
    ow, oh = round(w * _scale_rate), round(h * _scale_rate)
    scale_rate = float(ow) / w # err +-1
    img = img.resize((ow, oh), Image.BILINEAR)
    if boxes is not None:
        boxes = boxes * torch.FloatTensor([
                        scale_rate, scale_rate, scale_rate, scale_rate])
    if masks is not None:
        masks = scipy.ndimage.zoom(masks, zoom=[1, scale_rate, scale_rate], order=0)
        masks = masks[:, :size, :size]
    if mode == 'TRAIN':
        max_ofst_h = size - oh
        max_ofst_w = size - ow
        ofst_h = random.randint(0, max_ofst_h)
        ofst_w = random.randint(0, max_ofst_w)
        img = img.crop((-ofst_w, -ofst_h, 
                    size-ofst_w, size-ofst_h))
        if boxes is not None:
            boxes += torch.FloatTensor([ofst_h, ofst_w, ofst_h, ofst_w])
        if masks is not None:
            masks_tmp = np.zeros((masks.shape[0], size, size))
            masks_tmp[:, ofst_h:ofst_h+masks.shape[1], 
                ofst_w:ofst_w+masks.shape[2]] = masks
            masks = masks_tmp
        location = torch.FloatTensor([ofst_h, ofst_w, 
                        ofst_h+oh-1, ofst_w+ow-1, scale_rate])
    else:
        img = img.crop((0, 0, size, size))
        if masks is not None:
            masks_tmp = np.zeros((masks.shape[0], size, size))
            masks_tmp[:, :masks.shape[1], :masks.shape[2]] = masks
            masks = masks_tmp
        location = torch.FloatTensor([0, 0, oh-1, ow-1, scale_rate])
    return img, location, boxes, masks


COLOR_TABLE = [
    (256,0,0), (0,256,0), (0,0,256), 
    (255,0,255), (255,106,106),(139,58,58),(205,51,51),
    (139,0,139),(139,0,0),(144,238,144),(0,139,139)
] * 100


def draw_bbox_text(drawObj, ymin, xmin, ymax, xmax, text, color, bd=1):
    drawObj.rectangle((xmin, ymin, xmax, ymin+bd), fill=color)
    drawObj.rectangle((xmin, ymax-bd, xmax, ymax), fill=color)
    drawObj.rectangle((xmin, ymin, xmin+bd, ymax), fill=color)
    drawObj.rectangle((xmax-bd, ymin, xmax, ymax), fill=color)
    strlen = len(text)
    drawObj.rectangle((xmin, ymin, xmin+strlen*6+5, ymin+12), fill=color)
    drawObj.text((xmin+3, ymin), text)


def show_instance(img, boxes, labels, masks=None, name_table=None, scores=None, 
                    file_name=None, matplotlib=False):
    '''
    img:      FloatTensor(3, H, W) or PIL
    boxes:    FloatTensor(N, 4)
    labels:   LongTensor(N) 0:bg
    masks:    FloatTensor(N, H, W) or None
    scores:   FloatTensor(N) or None
    file_name: 'out.bmp' or None
    '''
    if not isinstance(img, Image.Image):
        img = transforms.ToPILImage()(img)
    # sort
    hw = boxes[:, 2:] - boxes[:, :2]
    area = hw[:, 0] * hw[:, 1] # N
    select = area.sort(descending=True)[1] # L(N)
    # blend mask
    if masks is not None:
        img_mask = torch.zeros(3, masks.shape[1], masks.shape[2])
        for i in range(select.shape[0]):
            i = int(select[i])
            m = masks[i] == 1 # H,W
            color = COLOR_TABLE[i]
            img_mask[0, m] = color[0]
            img_mask[1, m] = color[1]
            img_mask[2, m] = color[2]
        img_mask = img_mask / 257.0
        img_mask = transforms.ToPILImage()(img_mask)
        img = Image.blend(img, img_mask, 0.4)
    # draw bbox
    drawObj = ImageDraw.Draw(img)
    for i in range(select.shape[0]):
        i = int(select[i])
        lb = int(labels[i])
        if lb > 0: # fg
            box = boxes[i]
            if scores is None:
                draw_bbox_text(drawObj, box[0], box[1], box[2], box[3], name_table[lb],
                    color=COLOR_TABLE[i])
            else:
                str_score = str(float(scores[i]))[:5]
                str_out = name_table[lb] + ': ' + str_score
                draw_bbox_text(drawObj, box[0], box[1], box[2], box[3], str_out, 
                    color=COLOR_TABLE[i])
    if file_name is not None:
        img.save(file_name)
    else:
        if matplotlib:
            plt.imshow(img, aspect='equal')
            plt.show()
        else: img.show()
