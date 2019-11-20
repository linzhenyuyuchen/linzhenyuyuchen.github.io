---
layout:     post
title:      KfbReader
subtitle:   KfbReader配置
date:       2019-11-06
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - KfbReader
---

# KfbReader

**Download Link**

`https://linzhenyuyuchen.github.io/files/kfbreader.zip`

1. 解压 ： kfbreader-linux 到项目目录下，并改文件夹名为 kfbreader_linux

2. 打开bash 进入 kfbreader_linux目录

3. 依次执行：

    `ln -s libhzzt.so libhzzt.so.1`

    `ln -s libopencv_world.so.3.4.5 libopencv_world.so.3.4`

4. 验证是否可以在python环境下import kfbReader
    python -c "import kfbReader"

**OR 需要在环境变量中加入当前路径：**

```
vim  ~/.bashrc
```

```
export  PYTHONPATH=your_kfb_path/kfbreader:$PYTHONPATH  其中，your_kfb_path需要修改为自己的路径
```

```
加入一行：export LD_LIBRARY_PATH=your_anaconda_path/anaconda3/lib:your_kfb_path/kfbreader:$LD_LIBRARY_PATH  其中，your_anaconda_path和your_kfb_path需要修改为自己的路径
```

```
source  ~/.bashrc
```

