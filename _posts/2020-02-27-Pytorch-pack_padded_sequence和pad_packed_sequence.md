---
layout:     post
title:      Pytorch pack_padded_sequence和pad_packed_sequence
subtitle:   Pytorch中的RNN之pack_padded_sequence()和pad_packed_sequence()
date:       2020-02-27
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - Pytorch
---

# torch.nn.utils.rnn.pack_padded_sequence()

将一个填充过的变长序列压紧

参数说明:

- input (Variable) – 变长序列 被填充后的 batch
- lengths (list[int]) – Variable 中 每个序列的长度。
- batch_first (bool, optional) – 如果是True，input的形状应该是B*T*size

# torch.nn.utils.rnn.pad_packed_sequence()

把压紧的序列再填充回来

参数说明:

- sequence (PackedSequence) – 将要被填充的 batch
- batch_first (bool, optional) – 如果为True，返回的数据的格式为 B×T×*。
