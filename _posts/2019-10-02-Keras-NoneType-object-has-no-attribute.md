---
layout:     post
title:      Keras NoneType object has no attribute image_data_format
subtitle:   AttributeError NoneType object has no attribute image_data_format
date:       2019-10-02
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Keras
---

Before:

```python
    resnext = ResNeXt101(
        include_top=False,
        weights='imagenet',
        input_shape=(PNG_SIZE,PNG_SIZE,3),
        pooling=None
    )
```

After:

```python
    resnext = ResNeXt101(
        include_top=False,
        weights='imagenet',
        input_shape=(PNG_SIZE,PNG_SIZE,3),
        pooling=None,
        backend=keras.backend,
        layers=keras.layers,
        models=keras.models,
        utils=keras.utils
    )
```


