---
layout:     post
title:      Python tqdm
subtitle:   tqdm
date:       2019-10-24
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Python
---

# tqdm

```python
for patient_id, group in tqdm(groups, total=len(groups)):
```

# 在notebook中使用

```python
from tqdm import tnrange, tqdm_notebook
with tqdm_notebook(total = len(ls), desc = 'round', leave = False) as pbar:
	char+=1
	pbar.set_description("Processing %s" % char)
    par.update(1)
```