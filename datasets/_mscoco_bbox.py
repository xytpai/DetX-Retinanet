import torch
import torchvision
import torchvision.transforms as transforms
import random
if __name__ != '__main__':
    from datasets.utils import *
else:
    from utils import *


class Dataset(torchvision.datasets.coco.CocoDetection):
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
        super(Dataset, self).__init__(root_img, file_json)
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
        # name_table
        self.index_to_coco = [i for i in range(len(self.name_table))]
        self.coco_to_index = {}
        for cate in self.coco.loadCats(self.coco.getCatIds()):
            name = cate['name']
            if name in self.name_table:
                index = self.name_table.index(name)
                self.index_to_coco[index] = cate['id']
                self.coco_to_index[cate['id']] = index
        # filter self.ids
        self.ids = sorted(self.ids)
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno, self.coco_to_index):
                ids.append(img_id)
        self.ids = ids
        
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
        img, anno = super(Dataset, self).__getitem__(idx)
        anno = [obj for obj in anno if obj['iscrowd'] == 0] # filter crowd annotations
        anno = [obj for obj in anno if obj['area'] > 0]
        anno = [obj for obj in anno if all(o > 2 for o in obj['bbox'][2:])]
        anno = [obj for obj in anno if obj['category_id'] in self.coco_to_index]
        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        xmin_ymin, w_h = boxes.split([2, 2], dim=1)
        xmax_ymax = xmin_ymin + w_h - 1
        xmin, ymin = xmin_ymin.split([1, 1], dim=1)
        xmax, ymax = xmax_ymax.split([1, 1], dim=1)
        boxes = torch.cat([ymin, xmin, ymax, xmax], dim=1)
        labels = [self.coco_to_index[obj['category_id']] for obj in anno]
        labels = torch.LongTensor(labels)
        # clamp
        boxes[:, :2].clamp_(min=0)
        boxes[:, 2].clamp_(max=float(img.size[1])-1)
        boxes[:, 3].clamp_(max=float(img.size[0])-1)
        # transform
        if random.random() < 0.5: img, boxes, _ = x_flip(img, boxes)
        img, location, boxes, _ = to_square(img, self.size, 
                                    self.transfer_p, self.transfer_min, boxes)
        img = transforms.ToTensor()(img)
        if self.normalize: img = self.normalizer(img)
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
