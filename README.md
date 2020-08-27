### DetX-RetinaNet

Implementation of RetinaNet in PyTorch. <br>
Focal Loss for Dense Object Detection. <br>
https://arxiv.org/abs/1708.02002 

#### Usage

```txt
1. Install
sh install.sh

2. Training COCO 1x
python tools/train.py --cfg configs/retinanet_r50_sq1025_1x.yaml

3. COCO Eval
python tools/eval_mscoco.py --cfg configs/retinanet_r50_sq1025_1x.yaml

4. Demo
python tools/demo.py --cfg configs/retinanet_r50_sq1025_1x.yaml
```
