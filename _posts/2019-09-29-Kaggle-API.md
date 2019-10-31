---
layout:     post
title:      Kaggle API
subtitle:   Kaggle 配置环境 下载数据集 提交
date:       2019-09-29
author:     LZY
header-img: img/lossfunction.jpeg
catalog: true
tags:
    - Kaggle
---

# Install

```
pip install --upgrade kaggle
```

Colab:

```
!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!kaggle -v
```

# Configure

```
mkdir -p ~/.kaggle
echo '{"username":"linzhenyu","key":"2c0efb5db836749c4d8"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

# List

```
kaggle competitions list -s health
```

# Download

下载 <competition_name>下的Data中某个文件 <filename>，指定下载路径<path>

```
kaggle competitions download -c <competition_name> -f <filename> -p <path>

-q, --quiet           Suppress printing information about the upload/download progress
```

```
kaggle competitions download -c rsna-intracranial-hemorrhage-detection
```

下载数据集

```
kaggle datasets download -d guiferviz/rsna_stage1_png_128
```

# Submit

```
usage: kaggle competitions submit [-h] [-c COMPETITION] -f FILE -m MESSAGE
                                  [-q]

required arguments:
  -f FILE, --file FILE  File for upload (full path)
  -m MESSAGE, --message MESSAGE
                        Message describing this submission

optional arguments:
  -h, --help            show this help message and exit
  -c COMPETITION, --competition COMPETITION
                        Competition URL suffix (use "kaggle competitions list" to show options)
                        If empty, the default competition will be used (use "kaggle config set competition")"
  -q, --quiet           Suppress printing information about download progress
```

```
kaggle competitions submit -c rsna-intracranial-hemorrhage-detection -f /disk/diskone/zylin/rsna128/n.csv -m densenet
```
