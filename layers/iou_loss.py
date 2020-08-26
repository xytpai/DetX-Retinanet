import torch
import torch.nn as nn 
import torch.nn.functional as F


class IOULoss(nn.Module):
    def __init__(self):
        super(IOULoss, self).__init__()

    def forward(self, bboxes1, bboxes2):
        # bboxes1:   FloatTensor(n, 4) # 4: ymin, xmin, ymax, xmax
        # bboxes2:   FloatTensor(n, 4)
        tl = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        br = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
        hw = (br - tl + 1).clamp(min=0)  # [rows, 2]
        overlap = hw[:, 0] * hw[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
        ious = (overlap + 1.0) / (area1 + area2 - overlap + 1.0)
        return (-ious.log()).sum()
