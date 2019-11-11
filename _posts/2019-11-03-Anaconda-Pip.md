---
layout:     post
title:      Anaconda Pip
subtitle:   anaconda pip 更换国内源
date:       2019-11-03
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Anaconda
    - Pip
---

[Reference](https://blog.csdn.net/qq_35860352/article/details/80207483)

# pip 暂时

`pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple`

# pip 永久

windows下，直接在user目录中创建一个pip目录，如：C:\Users\xx\pip，新建文件pip.ini，内容如下：

```
[global]  
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

# Problem

**安装库后pycharm中的项目仍无法使用该库**

1. 关闭项目

2. 删除Project目录下的`.idea`文件夹

3. 重启项目



# Anaconda 和 Pip 二者的区别

more