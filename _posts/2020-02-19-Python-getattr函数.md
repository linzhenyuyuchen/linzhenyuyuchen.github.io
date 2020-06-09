---
layout:     post
title:      Python getattr函数
subtitle:   Python getattr() 函数
date:       2020-02-19
author:     LZY
header-img: img/lossfunction.jpeg
catalog: true
tags:
    - Python
---

# getattr() 函数

> getattr() 函数用于返回一个对象属性值。

getattr(object, name[, default])

- object -- 对象。
- name -- 字符串，对象属性。
- default -- 默认返回值，如果不提供该参数，在没有对应属性时，将触发 AttributeError。


```python
>>> class A(object):
...     def set(self, a, b):
...         x = a
...         a = b
...         b = x
...         print a, b
... 
>>> a = A()                 
>>> c = getattr(a, 'set')
>>> c(a='1', b='2')
2 1
>>> 
```

```python
class A(object):
    self.res1 = 1
    self.res2 = func()

    def fuc():
        for i in range(2):
            res_name = f'res{i+1}'
            res = getattr(self,res_name)
```