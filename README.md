### DetX-Retinanet

Implementation of RetinaNet in PyTorch. <br>
Focal Loss for Dense Object Detection. <br>
https://arxiv.org/abs/1708.02002 

#### Performance

| mAP(This)~700px | mAP(Paper)-700px | Download | MD5                            |
| :-------------- | ---------------- | ----------------------------------- | :---------------------------------- |
| **35.9%**       | 35.1%            | [Baidu](https://pan.baidu.com/s/1EYhde3_qZX5GND79LW036A):9b04 |6f3a2f8f8493ca3ded665dc6917aa653|

```txt
iters: 87960  epoches: 12
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.359
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.545
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.382
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.195
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.398
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.478
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.307
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.571
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.660
```

#### Usage

```txt
1. Install (PyTorch >= 1.0.0)
sh install.sh

2. Training COCO 1x
python tools/train.py --cfg configs/retinanet_r50_sq1025_1x.yaml

3. COCO Eval
copy retinanet_r50_sq1025_1x.pkl to weights/
python tools/eval_mscoco.py --cfg configs/retinanet_r50_sq1025_1x.yaml

4. Demo
python tools/demo.py --cfg configs/retinanet_r50_sq1025_1x.yaml
```
