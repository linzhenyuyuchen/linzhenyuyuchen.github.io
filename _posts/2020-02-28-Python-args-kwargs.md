---
layout:     post
title:      Python args kwargs
subtitle:   参数传递 args kwargs
date:       2020-02-28
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Python
---

# 什么是*args和**kwargs

```python
def foo(*args, **kwargs):
    print 'args = ', args
    print 'kwargs = ', kwargs
    print '---------------------------------------'
 
if __name__ == '__main__':
    foo(1,2,3,4)
    foo(a=1,b=2,c=3)
    foo(1,2,3,4, a=1,b=2,c=3)
    foo('a', 1, None, a=1, b='2', c=3)

```

```
输出结果如下：
args =  (1, 2, 3, 4) 
kwargs =  {} 
--------------------------------------- 
args =  () 
kwargs =  {'a': 1, 'c': 3, 'b': 2} 
--------------------------------------- 
args =  (1, 2, 3, 4) 
kwargs =  {'a': 1, 'c': 3, 'b': 2} 
--------------------------------------- 
args =  ('a', 1, None) 
kwargs =  {'a': 1, 'c': 3, 'b': '2'} 
---------------------------------------
```

