---
layout:     post
title:      KDD RETAIN Pytorch
subtitle:   An Interpretable Predictive Model for
Healthcare using Reverse Time Attention Mechanism / Knowledge-Discovery in Databases
date:       2020-03-09
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - 数据挖掘
    - KDD
---

[Github](https://github.com/linzhenyuyuchen/RETAIN-Pytorch)

# 生成数据集

把process_mimic和csv文件放在同一个目录下，运行以下命令

```
python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv PATIENTS.csv 
```

".seqs" 文件包含每个病人的记录序列，每个记录包含多种diagnosis codes.

".morts" (list) 文件包含7537个病人，每个病人死亡标记序列( 0 or 1 ).

".3digitICD9.seqs" (list)-> (list) 文件包含7537个病人，每个病人有若干记录(list)，其中记录数目大于2的病人有2377个，其他数目都是等于2，另外记录数目大于3的病人有1035个

".3digitICD9.types" (dict) 文件包含942个codes type，942是整个数据集中3-digit ICD9 codes的数目

# RETAIN

```
python retain.py  /home/coco/retain_data/

```




