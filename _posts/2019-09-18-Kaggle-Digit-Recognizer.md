---
layout:     post
title:      Kaggle Digit Recognizer
subtitle:   手写数字识别
date:       2019-09-18
author:     LZY
header-img: img/kaggle_digital_number.png
catalog: true
tags:
    - Kaggle
---

# Digit Recognizer

>Learn computer vision fundamentals with the famous MNIST data

## Kaggle enviroment

```
!pip install kaggle
!kaggle -h
```

**Problem**: `OSError: Could not find kaggle.json. Make sure it's located in C:\Users\xiaokeai\.kaggle. Or use the environment method.`

**Solution**:Generally you should get this file first from our homepage www.kaggle.com -> Your Account -> Create New API token. This will download a ready-to-go JSON file to place in you [user-home]/.kaggle folder. If there is no .kaggle folder yet, please create it first.

Ensure kaggle.json is in the location ~/.kaggle/kaggle.json to use the API.

## Download Data

```
kaggle competitions download -c digit-recognizer
```

`Downloading digit-recognizer.zip to C:\Users\xiaokeai`

```
kaggle datasets download -d taindow/rsna-train-stage-1-images-png-224x
```