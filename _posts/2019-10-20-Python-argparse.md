---
layout:     post
title:      Python argparse
subtitle:   import argparse 读入命令行参数
date:       2019-10-20
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Python
    - argparse
---

# argparse

> 读入命令行参数

[doc](https://docs.python.org/2/library/argparse.html)

```python
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # 传递值
    parser.add_argument('--num', type=int, default=0, help = " number input")
    # 传递布尔值
    parser.add_argument('--whether', action="store_true", help = " number input")
    # 限制传递参数的内容
    parser.add_argument('--num', type=int, default=0,  choices=[0, 1, 2], help = " number input")

    # 缩写法
    parser.add_argument('-n', '--num', type=int, default=0, help = " number input")
    parser.add_argument('-w', '--whether', action="store_true", help = " number input")

    return parser.parse_args()
```

```python
args = get_args()

print(args.num)

```