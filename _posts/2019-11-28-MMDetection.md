---
layout:     post
title:      MMDetection
subtitle:   mmdetection is an open source object detection toolbox based on PyTorch.
date:       2019-11-28
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - 
---

[Reference](https://github.com/open-mmlab/mmdetection)

# MMDetection

## 安装

[Reference](https://github.com/open-mmlab/mmdetection/blob/master/docs/INSTALL.md)

在Anaconda环境下：

```
conda create -n mmdetection python=3.6
source activate mmdetection
conda install pytorch torchvision -c pytorch
```

安装mmdetection

```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install mmcv
python setup.py develop
```

## 数据集格式

```
mmdetection
├── mmdet 检测框架代码
├── tools 训练和测试代码
├── configs 各种模型参数配置文件
├── data 新建存放数据集的文件夹
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── train
│   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012

```

### VOC 转换为mmdetection格式

Convert PASCAL VOC annotations to mmdetection format

```
python tools/convert_datasets/pascal_voc.py /data1/lzy/VOCdevkit/ -o /data1/lzy/voc_mmdetection/
```

## 训练

[Model Zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/MODEL_ZOO.md)

```
cd mmdetection
python tools/train.py configs/ssd300_coco.py
```

## 测试

```
python tools/test.py configs/faster_rcnn_r50_fpn_1x.py \
    checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth \
    --show \
    --out results.pkl \
    --eval bbox segm
```

## 可视化

分类和回归的loss变化曲线，保存图到pdf文件
```

python tools/analyze_logs.py plot_curve xxx.log.json --keys loss_cls loss_reg --out losses.pdf

```


比较bbox mAP 

```
python tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
```

计算平均训练速度

```
python tools/analyze_logs.py cal_train_time work_dirs/ssd300_coco/20191217_132646.log.json
```

## 训练和测试自己的数据集

(1) 数据集为 COCO or PASCAL VOC 格式

复制mmdet/datasets/my_dataset.py重命名ead.py

```
class EadDataset(Dataset):


    CLASSES = ('specularity',
              'saturation',
              'artifact',
              'blur',
              'contrast',
              'bubbles',
              'instrument',
              'blood')

```

在`mmdet/datasets/__init__.py`添加一行

```
from .ead import EadDataset
```


(2) 自定义格式

The annotation of a dataset is a list of dict, each dict corresponds to an image.

数据集的标注是每个图片对应的dict组成的list

```
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        'ann': {
            'bboxes': <np.ndarray, float32> (n, 4),
            'labels': <np.ndarray, int64> (n, ),
            'bboxes_ignore': <np.ndarray, float32> (k, 4),
            'labels_ignore': <np.ndarray, int64> (k, ) (optional field)
        }
    },
    ...
]
```

Some datasets may provide annotations like `crowd/difficult/ignored bboxes`, we use `bboxes_ignore` and `labels_ignore` to cover them.

**There are two ways to work with custom datasets.**

1. online conversion

You can write a new Dataset class inherited from CustomDataset, and overwrite two methods load_annotations(self, ann_file) and get_ann_info(self, idx), like CocoDataset and VOCDataset.

2. offline conversion -> (3)

You can convert the annotation format to the expected format above and save it to a pickle or json file, like `tools/convert_datasets/pascal_voc.py`. Then you can simply use CustomDataset.


(3) 修改模型配置文件中的数据集路径

In `configs/faster_rcnn_r50_fpn_1x.py`

修改数据路径以及num_classes=类别数+1(背景)

```
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/ead_train.json',
        img_prefix=data_root + 'ead_train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/ead_val.json',
        img_prefix=data_root + 'ead_val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/ead_val.json',
        img_prefix=data_root + 'ead_val/',
        pipeline=test_pipeline))
```

In `mmdet/datasets/xml_style.py`

修改图片后缀名

```
    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = 'JPEGImages/{}.png'.format(img_id)
```
