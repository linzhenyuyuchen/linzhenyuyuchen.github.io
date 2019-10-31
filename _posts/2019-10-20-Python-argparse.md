---
layout:     post
title:      Python argparse
subtitle:   import argparse
date:       2019-10-20
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Python
    - argparse
---

# argparse

```python
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0)

    return parser.parse_args()
```

```python
args = get_args()

print(args.num)

```