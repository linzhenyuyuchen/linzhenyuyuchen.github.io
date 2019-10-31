---
layout:     post
title:      Python multiprocessing
subtitle:   from multiprocessing import Pool
date:       2019-10-19
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Python
    - multiprocessing
---

[Reference]()

# multiprocessing

```python
from multiprocessing import Pool

num = 4 # thread number
with Pool(processes = num) as pool:
    records = list(
        tqdm(
            iterable=pool.imap_unordered(func,data_input),
            total=len(data_input)
        )
    )

```
