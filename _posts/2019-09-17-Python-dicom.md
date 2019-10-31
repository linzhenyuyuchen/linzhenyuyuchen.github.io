---
layout:     post
title:      Python pydicom
subtitle:   pydicom command
date:       2019-09-17
author:     LZY
header-img: img/lossfunction.jpeg
catalog: true
tags:
    - Python
---

# pydicom

Medical images are stored in a special format known as DICOM files `(*.dcm)`.

They contain a combination of `header metadata` as well as underlying `raw image arrays for pixel data`.

In Python, one popular library to access and manipulate DICOM files is the `pydicom` module. 

```python
import pydicom
dcm_filename='/tmp/filename.dcm'
dcm_data=pydicom.read_file(dcm_filename)
print(dcm_data)
```

```
(0008, 0005) Specific Character Set              CS: 'ISO_IR 100'

...

(7fe0, 0010) Pixel Data                          OB: Array of 142006 bytes
```

Most of the standard headers containing patient identifable information have been anonymized (removed) so we are left with a relatively sparse set of metadata.

The primary field we will be accessing is the underlying pixel data as follows:

```python
im = dcm_data.pixel_array
print(type(im))
print(im.dtype)
print(im.shape)
```

```
<class 'numpy.ndarray'>
uint8
(1024, 1024)
```

As we can see here, the pixel array data is stored as `a Numpy array`, a powerful numeric Python library for handling and manipulating matrix data (among other things). 