---
layout:     post
title:      Pytorch 清除GPU存储
subtitle:   清除GPU存储
date:       2020-01-24
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - 深度学习
    - Pytorch
---

# 清除GPU存储

有时Control-C中止运行后GPU存储没有及时释放，需要手动清空。在PyTorch内部可以

```python
torch.cuda.empty_cache()
```

或在命令行可以先使用ps找到程序的PID，再使用kill结束该进程

```
ps aux | grep python
kill -9 [pid]
```


或者直接重置没有被清空的GPU


```
nvidia-smi --gpu-reset -i [gpu_id]
```

