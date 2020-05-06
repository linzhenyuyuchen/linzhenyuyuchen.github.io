---
layout:     post
title:      Tensor PIL ndarray 转换
subtitle:   np.ndarray torch.Tensor与PIL.Image转换
date:       2020-01-27
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - Python
---

# torch.Tensor与PIL.Image转换

PyTorch中的张量默认采用N×D×H×W的顺序，并且数据范围在[0, 1]，需要进行转置和规范化。

```python
# torch.Tensor -> PIL.Image.
image = PIL.Image.fromarray(torch.clamp(tensor * 255, min=0, max=255
    ).byte().permute(1, 2, 0).cpu().numpy())
image = torchvision.transforms.functional.to_pil_image(tensor)  # Equivalently way

# PIL.Image -> torch.Tensor.
tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))
    ).permute(2, 0, 1).float() / 255
tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path))  # Equivalently way
```

# np.ndarray与PIL.Image转换

```python
# np.ndarray -> PIL.Image.
image = PIL.Image.fromarray(ndarray.astype(np.uint8))

# PIL.Image -> np.ndarray.
ndarray = np.asarray(PIL.Image.open(path))
```


