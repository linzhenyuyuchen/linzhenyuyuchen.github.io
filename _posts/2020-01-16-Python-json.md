---
layout:     post
title:      Python json
subtitle:   Data pretty printer
date:       2020-01-16
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - json
    - Python
---

# 将数据保存为.json文件

```python
model={} #数据
with open("./hmm.json",'w',encoding='utf-8') as json_file:
    json.dump(model,json_file,ensure_ascii=False)
```

# 读取.json文件

```python
model={} #存放读取的数据
with open("./hmm.json",'r',encoding='utf-8') as json_file:
    model=json.load(json_file)
```
