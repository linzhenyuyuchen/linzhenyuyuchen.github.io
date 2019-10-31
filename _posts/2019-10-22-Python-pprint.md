---
layout:     post
title:      Python pprint
subtitle:   Data pretty printer
date:       2019-10-22
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - pprint
    - Python
---

[Reference](https://docs.python.org/2/library/pprint.html)

The pprint module provides a capability to `“pretty-print”` **arbitrary Python data structures** in a form which can be used as input to the interpreter. 

If the formatted structures include objects which are **not fundamental Python types**, the representation may not be loadable.

This may be the case if objects such as files, sockets, classes, or instances are included, as well as many other built-in objects which are not representable as Python constants.

# pprint

```python
from pprint import pprint

pprint(arbitrary_Python_data_structures)
```