---
layout:     post
title:      Python pickle
subtitle:   import pickle
date:       2019-10-18
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Python
    - pickle
---

```python
import pickle
```

# pickle load

```python
with open('file.pkl','rb') as f:
    data = pickle.load(f)
```

# pickle dump

```python
with open('file.pkl','wb') as f:
    pickle.dump(data,f)
```
