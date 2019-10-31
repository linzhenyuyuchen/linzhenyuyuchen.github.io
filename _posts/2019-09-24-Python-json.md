---
layout:     post
title:      Python json
subtitle:   json command
date:       2019-09-24
author:     LZY
header-img: img/lossfunction.jpeg
catalog: true
tags:
    - Python
---

# json

[Reference](https://python3-cookbook.readthedocs.io/zh_CN/latest/c06/p02_read-write_json_data.html)

```python
# Writing JSON data
with open('data.json', 'w') as f:
    json.dump(data, f)

# Reading data back
with open('data.json', 'r') as f:
    data = json.load(f)
```
