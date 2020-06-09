---
layout:     post
title:      pycocotools 修改标注框颜色
subtitle:   pycocotools 固定标注框颜色
date:       2020-02-01
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - pycocotools
---

# 修改标注框颜色

## 下载源代码

`https://pypi.org/project/pycocotools/#files`

## 修改文件

修改`pycocotools-2.0.0/pycocotools/coco.py` 的`showAnns`函数

```python
# 增加以下

import seaborn as sns
cp = sns.color_palette("bright",9)
# 生成的数量可根据实际类别数目而定

# 修改以下
for ann in anns:
    # c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
    # 替换为
    c = list(cp[ann["category_id"]])

```

## 安装

```
cd ~/pycocotools-2.0.0
pip install .
```


## 其他方法

直接修改安装的库文件

文件夹： `/usr/local/lib/python3.6/dist-packages/pycocotools`