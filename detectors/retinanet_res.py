import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from layers import *
# TODO: choose backbone
from detectors.backbones import *
from detectors.necks import *
from detectors.heads import *


def tlbr2yxyx_anchor_yxyx(reg, acr, factor=1.0):
    '''
    reg: F(..., 4)
    acr: F(..., 4) yxyx
    ->   F(..., 4)
    '''
    acr_hw = acr[..., 2:] - acr[..., :2] + 1
    ctr_yx = (acr[..., :2] + acr[..., 2:])/2.0
    ymin_xmin = ctr_yx + reg[..., :2]*acr_hw*factor
    ymax_xmax = ctr_yx + reg[..., 2:]*acr_hw*factor
    return torch.cat([ymin_xmin, ymax_xmax], dim=-1)


class Detector(nn.Module):
    def __init__(self, cfg, mode='TEST'):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.register_buffer('trained_log', torch.zeros(2).long())
        self.register_buffer('anchors', torch.FloatTensor([
            [32.00, 32.00],
            [40.32, 40.32],
            [50.80, 50.80],
            [22.63, 45.25],
            [28.51, 57.02],
            [35.92, 71.84],
            [45.25, 22.63],
            [57.02, 28.51],
            [71.84, 35.92],
        ]))
        self.iou_th     = (0.4, 0.5)
        self.num_class  = self.cfg['DETECTOR']['NUM_CLASS']
        self.backbone   = ResNet(self.cfg['DETECTOR']['DEPTH'])
        self.neck       = FPN(self.backbone.out_channels, 256)
        self.head       = RetinaNetHead(256, self.num_class, self.anchors.shape[0])
        if self.mode == 'TRAIN' and self.cfg['TRAIN']['PRETRAINED']:
            self.backbone.load_pretrained_params()
        # loss 
        self.sigmoid_focal_loss = SigmoidFocalLoss(2.0, 0.25)
        self.iou_loss = IOULoss()

    def forward(self, imgs, locations, label_cls=None, label_reg=None):
        '''
        imgs:       F(b, 3, size, size)
        locations:  F(b, 5)
        label_cls:  L(b, n)       0:pad
        label_reg:  F(b, n, 4)
        '''
        batch_size, channels, im_h, im_w = imgs.shape
        out = self.neck(self.backbone(imgs))
        layer_nums = []
        pred_cls, pred_reg, pred_acr = [], [], []
        for s, feature in enumerate(out):
            h_s, w_s = feature.shape[2], feature.shape[3]
            stride = (im_h-1) // (h_s-1)
            cls_s, reg_s = self.head(feature)
            cls_s = cls_s.permute(0,2,3,1).contiguous()
            reg_s = reg_s.permute(0,2,3,1).contiguous()
            cls_s = cls_s.view(batch_size, -1, self.num_class)
            reg_s = reg_s.view(batch_size, -1, 4)
            acr_s = anchor_scatter(self.anchors*(2**s), batch_size, h_s, w_s, stride) 
            # F(b, an, 4)
            reg_s = tlbr2yxyx_anchor_yxyx(reg_s, acr_s)
            pred_cls.append(cls_s)
            pred_reg.append(reg_s)
            pred_acr.append(acr_s)
            layer_nums.append(h_s*w_s*self.anchors.shape[0])
        pred_cls = torch.cat(pred_cls, dim=1) # F(b, an, num_class)
        pred_reg = torch.cat(pred_reg, dim=1) # F(b, an, 4) yxyx
        pred_acr = torch.cat(pred_acr, dim=1) # F(b, an, 4) yxyx
        if (label_cls is not None) and (label_reg is not None):
            return self._loss(locations, pred_cls, pred_reg, pred_acr, label_cls, label_reg)
        else:
            return self._pred(locations, pred_cls, pred_reg, layer_nums, im_h, im_w)
    
    def _loss(self, locations, pred_cls, pred_reg, pred_acr, label_cls, label_reg):
        loss = []
        an = pred_cls.shape[1]
        for b in range(label_cls.shape[0]):
            # filter out padding labels
            label_cls_b, label_reg_b = label_cls[b], label_reg[b]
            m = label_cls_b > 0
            label_cls_b = label_cls_b[m]
            label_reg_b = label_reg_b[m]
            # get loss
            iou = box_iou(pred_acr[b], label_reg_b)
            iou_max, iou_max_idx = torch.max(iou, dim=1) # F(an), L(an)
            _iou_max_idx = torch.argmax(iou, dim=0) # F(n), L(n)
            m_neg = iou_max <  self.iou_th[0] # B(an)
            m_neg[_iou_max_idx] = 0
            m_pos = iou_max >= self.iou_th[1] # B(an)
            m_pos[_iou_max_idx] = 1
            num_pos = float(m_pos.sum().clamp(min=1))
            m_negpos = m_neg | m_pos # B(an)
            pred_cls_selected = pred_cls[b][m_negpos] # F(n+-, num_class)
            label_cls_selected = label_cls_b[iou_max_idx[m_negpos]]
            label_cls_selected[m_neg[m_negpos]] = 0 # L(n+-)
            pred_reg_selected = pred_reg[b][m_pos] # F(n+, 4)
            label_reg_selected = label_reg_b[iou_max_idx[m_pos]]
            loss_cls = self.sigmoid_focal_loss(pred_cls_selected, label_cls_selected).view(1)
            loss_reg = self.iou_loss(pred_reg_selected, label_reg_selected).view(1)
            loss.append((loss_cls+loss_reg)/num_pos)
        return torch.cat(loss)

    def _pred(self, locations, pred_cls, pred_reg, layer_nums, im_h, im_w):
        '''
        pred_cls_i: L(n)
        pred_cls_p: F(n)
        pred_reg:   F(n, 4)
        '''
        assert self.mode != 'TRAIN'
        batch_size = pred_cls.shape[0]
        assert batch_size == 1
        pred_cls = pred_cls.squeeze(0)
        pred_reg = pred_reg.squeeze(0)
        pred_cls_p, pred_cls_i = torch.max(pred_cls.sigmoid(), dim=1)
        pred_cls_i = pred_cls_i + 1
        start = 0
        _pred_cls_i, _pred_cls_p, _pred_reg = [], [], []
        for num in layer_nums:
            p = pred_cls_p[start:start+num]
            nms_maxnum = min(int(self.cfg[self.mode]['NMS_TOPK_P']), num)
            select = torch.topk(p, nms_maxnum, largest=True, dim=0)[1]
            _pred_cls_i.append(pred_cls_i[start:start+num][select])
            _pred_cls_p.append(pred_cls_p[start:start+num][select])
            _pred_reg.append(pred_reg[start:start+num][select])
            start += num
        pred_cls_i = torch.cat(_pred_cls_i, dim=0)
        pred_cls_p = torch.cat(_pred_cls_p, dim=0)
        pred_reg = torch.cat(_pred_reg, dim=0)
        m = pred_cls_p > self.cfg[self.mode]['NMS_TH']
        pred_cls_i = pred_cls_i[m]
        pred_cls_p = pred_cls_p[m]
        pred_reg = pred_reg[m]
        pred_reg[:, 2].clamp_(max=locations[2])
        pred_reg[:, 3].clamp_(max=locations[3])
        pred_reg[:, 0::2] -= float(locations[0])
        pred_reg[:, 1::2] -= float(locations[1])
        pred_reg[:, :2].clamp_(min=0)
        pred_reg = pred_reg/float(locations[4])
        # nms for each class
        _pred_cls_i, _pred_cls_p, _pred_reg = [], [], []
        for cls_id in range(1, self.num_class+1):
            m = pred_cls_i == cls_id
            if int(m.sum()) == 0: continue
            pred_cls_i_id = pred_cls_i[m]
            pred_cls_p_id = pred_cls_p[m]
            pred_reg_id = pred_reg[m]
            keep = box_nms(pred_reg_id, pred_cls_p_id, self.cfg[self.mode]['NMS_IOU'])
            _pred_cls_i.append(pred_cls_i_id[keep])
            _pred_cls_p.append(pred_cls_p_id[keep])
            _pred_reg.append(pred_reg_id[keep])
        return torch.cat(_pred_cls_i, dim=0), \
                    torch.cat(_pred_cls_p, dim=0), \
                        torch.cat(_pred_reg, dim=0)
