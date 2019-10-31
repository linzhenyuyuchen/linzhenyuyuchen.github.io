---
layout:     post
title:      Python Pandas
subtitle:   Pandas 基本操作
date:       2019-09-15
author:     LZY
header-img: img/lossfunction.jpeg
catalog: true
tags:
    - Python
    - Pandas
---

# Pandas

```python
import pandas as pd
```

## 输入/输出

读写CSV文件

[io reference](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table)

```python
df = pd.read_csv('filename.csv')

df.to_csv('filename.csv')
```

```python
df.head() #Return the last n rows (default 5)
df.tail()
```

数据解析

```python
def parse_data(df):
    parsed = {}
    extract_box = lambda row :[ row[''],row[''] ... ]
    for n,row in df.iterrows():
        pid = row['patiendID']
        if pid not in parsed:
            parsed[pid] = {
                '':'',
                '':row[''],
                'boxes':[]
            }
        if parsed[pid]['label'] = 1:
            parsed[pid]['boxes'].append(extract_box(row))
    return parse
```

iloc使用整数列表提取行，返回DataFrame类型。

```python
print(type(df.iloc[[0]]))
print(df.iloc[[0]])
print(df.iloc[[0, 1]])
```
