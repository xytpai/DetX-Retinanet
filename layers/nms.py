import torch
import torch.nn as nn 
import torch.nn.functional as F
import nms_cuda


def box_nms(bboxes, scores, threshold=0.5):
    '''
    Param:
    bboxes: FloatTensor(n,4) # 4: ymin, xmin, ymax, xmax
    scores: FloatTensor(n)

    Return:
    keep:   LongTensor(s)
    '''
    if bboxes.shape[0] == 0:
        return torch.zeros(0).long().to(bboxes.device)
    scores = scores.view(-1, 1)
    bboxes_scores = torch.cat([bboxes, scores], dim=1) # (n, 5)
    keep = nms_cuda.nms(bboxes_scores, threshold) # (s)
    return keep
