---
layout:     post
title:      Docker MMDetection
subtitle:   Docker File 安装 MMDetection
date:       2020-01-05
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - Docker
    - MMDetection
---

# Docker Image

```
docker pull vistart/mmdetection

# or

docker build -t mmdetection docker/
```

# Save as Image

```
docker commit djsifh213jf newname
```

# Run

```
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data mmdetection

# or

docker run --runtime=nvidia -p 10023:22 -p 10024:8888 --shm-size=16g  -it -v /data2/lzy/pathology/:/home/coco mmdetection /bin/bash
```



启动sshd服务

```
/etc/init.d/ssh start
nohup jupyter notebook --allow-root >jupyter.txt &
```

# Train

```
CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py ./configs/cascade_rcnn_r50_fpn_1x.py --gpus 4
```

# Test

## 检测评价矩阵

（detection evaluation metrics）

```
Average Precision (AP):
	AP		% AP at IoU=.50:.05:.95 (primary challenge metric) 
	APIoU=.50	% AP at IoU=.50 (PASCAL VOC metric) 
	APIoU=.75	% AP at IoU=.75 (strict metric)
AP Across Scales:
	APsmall		% AP for small objects: area < 322 
	APmedium	% AP for medium objects: 322 < area < 962 
	APlarge		% AP for large objects: area > 962
Average Recall (AR):
	ARmax=1		% AR given 1 detection per image 
	ARmax=10	% AR given 10 detections per image 
	ARmax=100	% AR given 100 detections per image
AR Across Scales:
	ARsmall		% AR for small objects: area < 322 
	ARmedium	% AR for medium objects: 322 < area < 962 
	ARlarge		% AR for large objects: area > 962
```

AP 和 AR 一般是在多个 IoU(Intersection over Union) 值取平均值. 具体地，采用了 10 个 IoU阈值 0.50:0.05:0.95. 对比于传统的只计算单个 IoU 阈值(0.50)的指标，这是一种突破. 对多个 IoU 阈值求平均，能够使得目标检测器具有更好的定位位置.

Precision是在识别出来的图片中，True positives所占的比率：Precision = tp / (tp+fp) 

Recall 是被正确识别出来目标的个数与测试集中所有目标个数的比值：Recall = tp / (tp+fn) 

Area COCO数据集中小目标物体数量比大目标物体更多. 具体地，标注的约有 41% 的目标物体是都很小的(small, 面积< 32x32=1024)，约有  34% 的目标物体是中等的(medium,  1024=32x32 < 面积 < 96x96=9216)，约有 24% 的目标物体是大的(large, 面积 > 96x96=9216). 面积(area) 是指 segmentation mask 中像素的数量.

## Analyze logs

Examples:

- Plot the classification loss of some run.

`python tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls`

- Plot the classification and regression loss of some run, and save the figure to a pdf.

`python tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_reg --out losses.pdf`

- Compare the bbox mAP of two runs in the same figure.

`python tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2`





