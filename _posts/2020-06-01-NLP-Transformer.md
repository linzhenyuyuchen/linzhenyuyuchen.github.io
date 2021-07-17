---
layout:     post
title:      NLP Transformer
subtitle:   NLP Transformer
date:       2020-06-01
author:     LZY
header-img: img/bg-20210718.jpg
catalog: true
tags:
    - NLP
---

# NLP Transformer

![](/img/31a961ee467928619d14ar.jpg)


## Encoder

Encoder 由N个相同的layer组成，即上图左侧部分。

每个layer包括2个sub-layer: multi-head self-attention mechanism 和 fully connected feed-forward network。

每个sub-layer都有residual connection和normalization：


![](/img/8889.png)



![](/img/100006a6.jpg)

### multi-head self-attention mechanism

> 自注意力机制


在对每个单词编码时关注输入句子的其它单词。

`The animal didn’t cross the street because it was too tired`

对于上面的句子，当模型处理单词`it`的时候，自注意力机制会允许`it`和`animal`建立联系。

---

**step-1**

自注意力对输入向量分别与三个权重矩阵相乘，得到一个查询向量，一个键向量和一个值向量。

![](/img/q14930.jpg)


**step-2**

当计算`thinking`的自注意力向量时，我们需要输入句子中的每个单词对它进行打分，用以评判对其它词的重视度。

其它词的键向量与`thinking`的查询向量计算点积

![](/img/q15202.jpg)

Softmax分数决定了每个单词对`thinking`的贡献，其中dk是键向量维度

---

**step-3**

将每个值向量乘以softmax分数，再求和。

![](/img/q15564.jpg)


---


**矩阵运算**

![](/img/111q3.jpg)

![](/img/qddddv4.jpg)

---

**multi-head**

多个以上的运算，保持独立的查询/键/值矩阵，从而产生不同的结果，得到多个不同的Z矩阵。

![](/img/7009086.jpg)

---

![](/img/ef7009206.jpg)


### fully connected feed-forward network

> 前馈神经网络

非线性变换，每个位置i的单词对应的变换参数都完全一样，所以称为position-wise。

## Decoder

解码器中也有编码器的自注意力self-attention层和前馈feed-forward层。

除此之外，这两个层之间还有一个注意力层，用来关注输入句子的相关部分（和seq2seq模型的注意力作用相似）。解码器中的自注意力层表现的模式与编码器不同：在解码器中，自注意力层只被允许处理输出序列中更靠前的那些位置。在softmax步骤前，它会把后面的位置给隐去（把它们设为-inf）。


## Linear and Softmax

![](/img/qq324324.jpg)

