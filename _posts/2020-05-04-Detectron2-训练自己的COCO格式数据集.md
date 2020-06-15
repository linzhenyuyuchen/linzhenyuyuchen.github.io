---
layout:     post
title:      Detectron2 训练自己的COCO格式数据集
subtitle:   训练自己的COCO格式数据集
date:       2020-05-04
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - PyTorch
    - 网络模型
    - 图像分割
    - Detectron2
    - 目标检测
---

# 安装Detectron2

```
pip install cython
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install -i https://pypi.douban.com/simple/ pyyaml==5.1.1
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```

# 修改文件

Detectron2 安装目录

`/usr/local/lib/python3.6/dist-packages/detectron2`

---

修改`./data/datasets/builtin.py`

```python
# 增加以下代码片段

_PREDEFINED_SPLITS_COCO_MY = {}
_PREDEFINED_SPLITS_COCO_MY["my_dataset"] = {
    "PT_train": ("train2017", "annotations3/cocojson_13_train_3.json"),
    "PT_val": ("train2017", "annotations3/cocojson_13_val_3.json"),
}

def register_my_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO_MY.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_my_coco__meta(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )
```

```python
# 增加以下代码片段

_root = os.getenv("DETECTRON2_DATASETS", "/home/coco/") # 数据目录
register_my_coco(_root)
```

---

修改`./data/datasets/builtin_meta.py`

```python
# 增加以下代码片段
def _get_my_coco__meta():
    ret = {
        "thing_dataset_id_to_contiguous_id": {0:0,1:1}, # 新id对应于数据集annotation中的id
        "thing_classes": ["0","1"], # 类别名称
        "thing_colors": [[73, 77, 174],[122, 134, 12]], # 每个类别对应的标注颜色RGB
    }
    return ret
```

---

修改配置文件`xxx.yaml`

```python
DATASETS:
  TRAIN: ("PT_train",)
  TEST: ("PT_val",)
```
