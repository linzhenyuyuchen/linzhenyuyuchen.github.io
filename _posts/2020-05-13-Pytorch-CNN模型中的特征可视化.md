---
layout:     post
title:      Pytorch CNN模型中的特征可视化
subtitle:   CNN模型中的特征可视化
date:       2020-05-13
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - Pytorch
---

# 方法1 grad-cam

[Grad-cam](https://linzhenyuyuchen.github.io/2020/01/31/Pytorch-Grad-CAM/)

# 方法2 特征图叠加

```python
import matplotlib.pyplot as plt
from pylab import *

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch):
    feature_map = img_batch
    #print(feature_map.shape)
 
    feature_map_combination = []
    #plt.figure()
 
    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)
    
    #num_pic = min(num_pic,20)
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        #plt.subplot(row, col, i + 1)
        #plt.imshow(feature_map_split)
        #axis('off')
    #plt.savefig('feature_map.png')
    #plt.show()
 
    # 各个特征图叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    return feature_map_sum
    #plt.imshow(feature_map_sum)
    #plt.savefig("feature_map_sum.png")
 
plt.figure(figsize=(10, 10))
for ii in range(len(img_list)):
    iii = img_list[ii]
    jjj = iii.replace(".png",".bmp").replace("pre_image","final_annotations_bmp")
    img_jjj = cv2.imread(jjj)
    plt.subplot(len(img_list), 3, ii*3+1)
    plt.imshow(img_jjj)
    # mask图

    img = cv2.imread(iii)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    inputs = torch.Tensor(img).permute(2,0,1).unsqueeze(0).cuda()
    outputs = model.senet(inputs)
    out_label = model(inputs)
    # predicted label
    print(1-np.argmax(out_label.cpu().detach().numpy()))
    feature = outputs.reshape(outputs.shape[1:])
    feature = feature.permute(1,2,0)
    heatmap = visualize_feature_map(feature.cpu().detach().numpy())
    plt.subplot(len(img_list), 3, ii*3+2)
    plt.imshow(heatmap)
    # heatmap图

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = superimposed_img  / superimposed_img.max()
    plt.subplot(len(img_list), 3, ii*3+3)
    plt.imshow(superimposed_img)
    # 将热图和原图重叠在一起
```

![](/img/index1231.png)

![](/img/index1232.png)


